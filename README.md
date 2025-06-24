# Compressing ECG data with transmission over a network in real-time 

This project aims to provide tools and methods for efficiently compressing ECG data for real-time network transmission. 

## How It Works

This project compresses real-time ECG data by first applying a simple delta encoding followed by byte compression using `zstandard`.
Multiple encoding mechanisms have been tested and the current pipeline has yielded the best results. Adaptive coding mechanisms like adaptive delta coding in the first phase yielded even better results, but adaptive coding is not viable in a real-time environment. `zstandard` internally uses a modified LZ77 compression technique and it works efficiently because of the repetitive nature of ECG data.

This mechanism is particularly useful for real-time applications where bandwidth is limited and low latency is crucial.

The current implementation simulates a network transmission by writing the compressed data to a file, which can be replaced with actual network transmission code in a real-world application.

The algorithm has been tested with multiple standard ECG datasets and has shown significant compression ratios while maintaining the integrity of the data, with error rates less than `1e-6`. It also comes with a CRC checksum to ensure data integrity during transmission.

## Dataset Compression Results

For representation purposes, one file from a each dataset is compressed and the results are shown below:

__**MIT-BIH Arrhythmia Database**__

```
Original size: 9.92 MB
Compressed size: 1.14 MB
Compression ratio: 8.701
```

__**Long Term ST Database**__

```
Original size: 314.25 MB
Compressed size: 35.54 MB
Compression ratio: 8.843
```

__**Motion Artifact Contaminated ECG Database**__

```
Original size: 0.12 MB
Compressed size: 0.01 MB
Compression ratio: 9.789
```

## Loading a dataset 

Uses `wfdb` to load Physionet databases currently. The ingestion logic can be changed to fit use cases.

## Contributions

All contributions are welcome. This project is licensed under MIT.
