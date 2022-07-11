# DMVER
This project is a benchmark for a new dataset DMVER.

DMVER: a Large Dataset of Micro-Videos for Emotion Recognition.

![See
data/Fig1(f).jpg](data/Fig1(f).jpg "data/Fig1(f).jpg")

This dataset was proposed for affective video content analysis(AVCA). It consists of 31,761 videos uploaded by more than 30,000 users on Tiktok, and the amount of data in each category is evenly distributed. This dataset is currently the largest dataset of videos for emotion recognition, including more data samples than other datasets

DMVER is a dataset with multidimensional information. In DMVER, we collect video data and video-related text information. And the audio and frame sequences can be extracted from video, and based on the frame sequences can compute optical flow sequences.

Description                            | Statistics 
-------------------------------------- | :-----------------:
Total number of micro-videos           |  31,761
Total number of text datas             | 27,010
Total number of video publishers       | 30,000
Total number of distinct topics        | 2
Total number of emotion classes        | 3
Minimum resolution of videos           | 720x560p
Max resolution of videos               | 1920x1080p
Average length of sentences in seconds | 15
 
We use the current mainstream emotion recognition methods in videos to define the baseline of the proposed dataset. We analyze the factors that affect the performance of the model based on the results. The research results prove the usability and challenge of the dataset. It may bring some new inspirations to future correlations.
