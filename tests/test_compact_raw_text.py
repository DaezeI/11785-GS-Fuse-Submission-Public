import io
import json
import unittest

from model.compact_news_vocab import PAD_ID, UNK_ID, TrainVocabulary

try:
    import torch
except ModuleNotFoundError as err:
    if err.name != "torch":
        raise
    torch = None
else:
    from model.compact_raw_text import RawTextEncoder


class VocabularyTest(unittest.TestCase):
    def test_train_only_and_reserved_ids(self):
        vocab = TrainVocabulary.fit(["price PRICE hike", "hike eased"], max_vocab=5, max_tokens=4)
        ids = vocab.encode("validationOnly price")
        self.assertEqual(ids, [UNK_ID, 3, PAD_ID, PAD_ID])
        self.assertEqual(vocab.encode("  ..."), [PAD_ID] * 4)
        self.assertEqual(len(vocab), 5)
        vocab.encode("validationOnly testOnly")
        self.assertEqual(vocab.to_dict()["words"], ["hike", "price", "eased"])

    def test_order_cap_and_normalization(self):
        articles = ["ＦＥＤ rate hidden", "rate target"]
        vocab = TrainVocabulary.fit(iter(articles), max_vocab=4, max_tokens=2)
        self.assertEqual(vocab.to_dict(), TrainVocabulary.fit(reversed(articles), 4, 2).to_dict())
        self.assertEqual(vocab.to_dict()["words"], ["rate", "fed"])
        self.assertEqual(vocab.encode("FED rate never-read"), [3, 2])

    def test_json_round_trip(self):
        vocab = TrainVocabulary.fit(["Fed's rates", "price"])
        restored = TrainVocabulary.from_dict(json.loads(json.dumps(vocab.to_dict())))
        self.assertEqual(vocab.to_dict(), restored.to_dict())
        self.assertEqual(vocab.encode("FED’S rates tomorrow"), restored.encode("FED’S rates tomorrow"))
        with self.assertRaises(ValueError):
            TrainVocabulary.from_dict({**vocab.to_dict(), "version": "unknown"})


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class EncoderTest(unittest.TestCase):
    def test_padding_and_empty_are_inert(self):
        encoder = RawTextEncoder(8, 4, embedding_dim=3)
        with torch.no_grad():
            # Masking still works if a checkpoint supplied a non-zero pad row.
            encoder.embedding.weight[PAD_ID].fill_(50)
        short = encoder(torch.tensor([[2, 3]]))
        mixed = encoder(torch.tensor([[2, 3, 0, 0], [0, 0, 0, 0]]))
        torch.testing.assert_close(short[0], mixed[0])
        torch.testing.assert_close(mixed[1], torch.zeros(4), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            encoder(torch.tensor([2, 3]))

    def test_forecast_residual_gradient_with_frozen_price(self):
        torch.manual_seed(7)
        encoder = RawTextEncoder(8, 3, embedding_dim=4)
        price_reference = torch.nn.Linear(4, 3).requires_grad_(False)
        reference = price_reference(torch.ones((2, 4)))
        pred = reference + encoder(torch.tensor([[2, 1, 0], [3, 2, 0]]))
        before = encoder.embedding.weight.detach().clone()
        opt = torch.optim.SGD(encoder.parameters(), lr=0.1)
        (pred - 1.0).square().mean().backward()
        self.assertGreater(encoder.embedding.weight.grad[2].abs().sum().item(), 0)
        self.assertGreater(encoder.embedding.weight.grad[UNK_ID].abs().sum().item(), 0)
        self.assertGreater(encoder.projection.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(encoder.embedding.weight.grad[PAD_ID], torch.zeros(4))
        self.assertTrue(all(p.grad is None for p in price_reference.parameters()))
        opt.step()
        self.assertFalse(torch.equal(before[2], encoder.embedding.weight[2]))

    def test_weights_and_vocabulary_round_trip(self):
        vocab = TrainVocabulary.fit(["one two"], max_tokens=4)
        encoder = RawTextEncoder(len(vocab), 3, embedding_dim=4)
        b = io.BytesIO()
        torch.save({"tokenizer": vocab.to_dict(), "weights": encoder.state_dict()}, b)
        b.seek(0)
        readback = torch.load(b, weights_only=True)
        new_vocab = TrainVocabulary.from_dict(readback["tokenizer"])
        new_encoder = RawTextEncoder(len(new_vocab), 3, embedding_dim=4)
        new_encoder.load_state_dict(readback["weights"])
        torch.testing.assert_close(
            encoder(torch.tensor([vocab.encode("one two three")])),
            new_encoder(torch.tensor([new_vocab.encode("one two three")])),
        )


if __name__ == "__main__":
    unittest.main()
