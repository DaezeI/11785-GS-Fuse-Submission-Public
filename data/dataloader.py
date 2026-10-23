from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import random
from torch import from_numpy
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
from collections import defaultdict
import glob
class data_set(Dataset):
    def __init__(self, list):
        self.list = list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, id):
        file, seq_data, pred_data, scale_seq_data, scaled_pred_data = self.list[id]
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return text, seq_data.T, pred_data.T, scale_seq_data.T, scaled_pred_data.T

class data_set_new(Dataset):
    def __init__(self, list):
        self.list = list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, id):
        item = self.list[id]
        # Support both 7-element (legacy) and 8-element (with event_type) formats; always return 7 for backward compat
        if len(item) == 8:
            full_summary_file, sent_reports_files, negative_samples_on_type_files, seq_data, pred_data, event_type, scale_seq_data, scaled_pred_data = item
        else:
            full_summary_file, sent_reports_files, negative_samples_on_type_files, seq_data, pred_data, scale_seq_data, scaled_pred_data = item
        
        with open(full_summary_file, 'r', encoding='utf-8', errors='ignore') as f:
            full_summary = f.read()
        sent_reports = []
        
        for sent_report in sent_reports_files:
            with open(sent_report, 'r', encoding='utf-8', errors='ignore') as f:
                sent_reports.append(f.read())
        negative_sample_reports_on_type = []
        
        for negative_samples_on_type_file in negative_samples_on_type_files:
            with open(negative_samples_on_type_file, 'r', encoding='utf-8', errors='ignore') as f:
                negative_sample_reports_on_type.append(f.read())
        return full_summary, sent_reports, negative_sample_reports_on_type, seq_data.T, pred_data.T, scale_seq_data.T, scaled_pred_data.T


class event_set(object):

    def __init__(self,
                 seq_len,
                 pred_len,

                 event_id=0,
                 series_id='SP500',

                 shuffle=False,
                 batch_size=32,
                 event_dir='data/event',
                 series_dir='data/series',
                 scale=True,
                 split_by_time=False,
                 num_workers=0,
                 ):

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_workers = num_workers

        # event_dir = os.path.join('data', 'event')
        # series_dir = os.path.join('data', 'series')
        event_dirs = []

        # Find all event subfolders and store directories
        # Sorted: the train/vali/test split is positional, so raw listdir order would make it OS-dependent.
        for main_folder in sorted(os.listdir(event_dir)):
            main_folder_path = os.path.join(event_dir, main_folder)
            if os.path.isdir(main_folder_path):
                for sub_folder in sorted(os.listdir(main_folder_path)):
                    sub_folder_path = os.path.join(main_folder_path, sub_folder)
                    if os.path.isdir(sub_folder_path):
                        event_dirs.append(sub_folder_path)

        # Now event_dirs contains paths to all event subdirectories
        series_file = None
        for file in os.listdir(series_dir):
            if file.split('.')[0] == series_id:
                series_file = os.path.join(series_dir, file)
                break

        if series_file is None:
            print('No time series data found')
            exit(0)
        else:
            raw_df = pd.read_csv(series_file)

        # Get date_int list
        self.dates = raw_df['date_int'].tolist()

        # Pre-cache event subfolder dates to avoid tens of thousands of redundant disk scans on Windows
        self._cached_event_subfolders = {}
        for etype in ["1", "2", "3", "4", "5", "6"]:
            etype_folder = os.path.join(event_dir, etype)
            if os.path.isdir(etype_folder):
                subf = sorted([f for f in os.listdir(etype_folder) if os.path.isdir(os.path.join(etype_folder, f))])
                self._cached_event_subfolders[etype] = (etype_folder, subf)

        event_files = []
        for event_folder in event_dirs:
            folder_files = sorted(os.listdir(event_folder))
            event_type = os.path.basename(os.path.dirname(event_folder))
            for file in folder_files:
                if file.endswith('txt_full_summary.txt'):
                    # This is the main file to process
                    date = file.split('.')[0]
                    if event_id == 4:
                        date += '1400005'
                    else:
                        date += '083000'

                    if int(date) > self.dates[0] and int(date) < self.dates[-1]:
                        full_summary_path = os.path.join(event_folder, file)

                        # Now gather the sent report files from 0 to 10 from in-memory folder files
                        sent_reports = []
                        for i in range(11):  # from sent0 to sent10
                            suffix = f".txt_report_sent{i}.txt"
                            matching = [f for f in folder_files if f.endswith(suffix)]
                            if matching:
                                sent_path = os.path.join(event_folder, matching[-1])
                                sent_reports.append(sent_path)
                        if len(sent_reports) != 10:
                            sent_reports = sent_reports[:10]
                        nagative_samples_on_type = self.negative_event_based_on_type(date, event_type, event_folder, n=5)
                        # Append the main summary and sent reports together (event_type for gate-openness experiments)
                        event_files.append([date, full_summary_path, sent_reports, nagative_samples_on_type, event_type])

        if len(event_files) < 10:
            print('Not enough event data found')
            exit(0)

        # If split_by_time: sort events chronologically so train/vali/test are first/mid/last in time
        if split_by_time:
            event_files.sort(key=lambda x: int(x[0]))

        # Process the time series data and append event information
        self.data = []
        for date, full_summary, sent_reports, nagative_samples_on_type, event_type in event_files:
            id = self.b_search(int(date), self.dates)
            if id < seq_len - 1 or id + pred_len - 1 > len(self.dates):
                continue
            else:
                seq_data = from_numpy(
                    raw_df.iloc[id - seq_len + 1: id + 1].drop(columns=['date', 'date_int']).to_numpy())
                pred_data = from_numpy(
                    raw_df.iloc[id + 1:id + pred_len + 1].drop(columns=['date', 'date_int']).to_numpy())
                self.data.append([full_summary, sent_reports, nagative_samples_on_type, seq_data, pred_data, event_type])

        # When splitting by time, do not shuffle so that borders are chronological (first 60% / next 20% / last 20%)
        if not split_by_time and shuffle:
            random.shuffle(self.data)

        self.d = raw_df.shape[1] - 2
        ratio = [0.6, 0.2, 0.2]  # train : test : vali
        n = len(self.data)
        border1 = [0, int(ratio[0] * n), int((ratio[0] + ratio[1]) * n)]
        border2 = [int(ratio[0] * n), int((ratio[0] + ratio[1]) * n), n]

        self.scaler = StandardScaler()
        if scale:
            train_seq_data = np.concatenate([i[3].numpy() for i in self.data[border1[0]:border2[0]]] +
                                            [i[4].numpy() for i in self.data[border1[0]:border2[0]]], axis=0)
            self.scaler.fit(train_seq_data)
            for data in self.data:
                data.append(from_numpy(self.scaler.transform(data[3])))
                data.append(from_numpy(self.scaler.transform(data[4])))

        # self.data: [main_file, sent_reports, seq_data, pred_data, scale_seq, scale_pred]
        self.train_set, self.train_loader = self.get_data(self.data[border1[0]: border2[0]], batch_size, self.num_workers)
        self.test_set, self.test_loader = self.get_data(self.data[border1[1]: border2[1]], batch_size, self.num_workers)
        self.vali_set, self.vali_loader = self.get_data(self.data[border1[2]: border2[2]], batch_size, self.num_workers)
        split_mode = "by time (6:2:2)" if split_by_time else "by index (6:2:2, shuffle={})".format(shuffle)
        print(f"Data Loaded: TRAIN: {len(self.train_set)}, VALI: {len(self.vali_set)}, TEST: {len(self.test_set)} [split: {split_mode}]")



    def negative_event_based_on_type(self, date, current_event_type, event_folder, n=5):
        date = date[:8]  # Trim the date to the first 8 characters (if needed)
        event_list = ["1", "2", "3", "4", "5", "6"]
        event_files = defaultdict(list)
        target_date = datetime.strptime(date, "%Y%m%d")

        for event_type in event_list:
            if event_type == current_event_type:
                continue
            cached = self._cached_event_subfolders.get(event_type)
            if not cached or not cached[1]:
                continue
            folder, sorted_subf = cached
            closest_folders = self.find_closest_n_dates_from_list(target_date, sorted_subf, n)
            for closest_folder in closest_folders:
                if closest_folder:
                    summary_file = os.path.join(folder, closest_folder, f"{closest_folder}_full_summary.txt")
                    event_files[event_type].append(summary_file)

        # Randomly sample `n` files from the collected files, ensuring balance between event types
        summary_files = []
        event_types_with_files = list(event_files.keys())

        # Ensure balanced sampling across event types
        while len(summary_files) < n and event_types_with_files:
            random.shuffle(event_types_with_files)

            for event_type in event_types_with_files[:]:
                if event_files[event_type]:
                    summary_files.append(event_files[event_type].pop())
                    if len(summary_files) == n:
                        break
                if not event_files[event_type]:
                    event_types_with_files.remove(event_type)
        return summary_files

    def find_closest_n_dates_from_list(self, target_date, sorted_dates, n=6):
        if not sorted_dates:
            return []
        target_date_str = target_date.strftime("%Y%m%d")
        closest_index = self.b_search(target_date_str, sorted_dates)
        if closest_index == -1:
            closest_index = 0

        left = closest_index - 1
        right = closest_index + 1
        left_dates = []
        right_dates = []

        while len(left_dates) < n // 2 and left >= 0:
            left_dates.append(sorted_dates[left])
            left -= 1

        while len(right_dates) < n // 2 and right < len(sorted_dates):
            right_dates.append(sorted_dates[right])
            right += 1

        while len(left_dates) + len(right_dates) < n and left >= 0:
            left_dates.append(sorted_dates[left])
            left -= 1

        while len(left_dates) + len(right_dates) < n and right < len(sorted_dates):
            right_dates.append(sorted_dates[right])
            right += 1

        return sorted(left_dates + right_dates)

    def b_search(self, date, dates):
            i, j = 0, len(dates) - 1
            while i <= j:
                mid = (i + j) // 2
                if mid >= len(dates) - 1:
                    break
                if date >= dates[mid] and date < dates[mid + 1]:
                    return mid
                else:
                    if date < dates[mid]:
                        j = mid - 1
                    if date >= dates[mid + 1]:
                        i = mid + 1
            return -1

    def find_closest_n_dates(self, target_date, event_folder, n=6):
        """
        Find exactly `n` closest dates before and after the target date, excluding the target date itself.
        :param target_date: The date to search for (datetime object)
        :param event_folder: Path to the event folder
        :param n: Number of closest dates to find in total, excluding the target date
        :return: List of closest date subfolders (balanced between before and after)
        """
        # Get a list of all subfolders (which are dates) in the current event folder
        subfolders = [f for f in os.listdir(event_folder) if os.path.isdir(os.path.join(event_folder, f))]

        if not subfolders:
            return []

        # Sort the subfolder names assuming they are valid dates
        sorted_dates = sorted(subfolders)

        # Convert the target date to a string for comparison
        target_date_str = target_date.strftime("%Y%m%d")

        # Use the binary search function to find the closest date
        closest_index = self.b_search(target_date_str, sorted_dates)

        # Initialize left and right pointers
        left = closest_index - 1
        right = closest_index + 1

        # Initialize lists to store left and right dates
        left_dates = []
        right_dates = []

        # Collect up to `n // 2` dates from the left (before the target date)
        while len(left_dates) < n // 2 and left >= 0:
            left_dates.append(sorted_dates[left])
            left -= 1

        # Collect up to `n // 2` dates from the right (after the target date)
        while len(right_dates) < n // 2 and right < len(sorted_dates):
            right_dates.append(sorted_dates[right])
            right += 1

        # If there are fewer dates on one side, fill the remaining from the other side
        while len(left_dates) + len(right_dates) < n and left >= 0:
            left_dates.append(sorted_dates[left])
            left -= 1

        while len(left_dates) + len(right_dates) < n and right < len(sorted_dates):
            right_dates.append(sorted_dates[right])
            right += 1

        # Combine left and right dates and return sorted
        closest_dates = sorted(left_dates + right_dates)

        return closest_dates

    def get_data(self, data, batch_size, num_workers=15):
        dataset = data_set_new(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        return dataset, dataloader
