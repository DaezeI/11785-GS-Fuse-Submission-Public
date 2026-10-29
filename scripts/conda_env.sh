# Sourced by the other scripts: makes `conda activate` work in non-interactive shells.
for base in "$(conda info --base 2>/dev/null)" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
  if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
    source "$base/etc/profile.d/conda.sh"
    return 0
  fi
done
echo "conda not found" >&2
return 1
