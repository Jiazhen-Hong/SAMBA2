# SAMBA-T

Paper-aligned reproduction using the TUAB BIOT train, validation, and test recording split. Each example is a non-overlapping 100-second window with 16 bipolar channels at 128 Hz.

Expected release directory: `checkpoints/SAMBA-T/`.

The preprocessing and training command will be added only after the retained TUAB source format is verified against the original EDF-to-bipolar preprocessing code. The release must resolve exactly 39,810 windows; an unverified data conversion must not be used for this checkpoint.
