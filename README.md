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

The performance of current mainstream emotion recognition methods in videos of the proposed dataset. And the existing dataset is introduced for performance comparison.

<table border="1" cellspacing="1">
<tr>
<td rowspan="2" align="center">Method</td>
<td colspan="3" align="center">IEMOCAP</td>
<td colspan="3" align="center">Music-video</td>
<td colspan="3" align="center">DMVER</td>
</tr>
<tr>
<td>RGB</td>
<td>Flow/Audio</td>
<td>Joint</td>
<td>RGB</td>
<td>Flow/Audio</td>
<td>Joint</td>
<td>RGB</td>
<td>Flow/Audio</td>
<td>Joint</td>
</tr>
<tr>
<td>ConvNet+LSTM</td>
<td>50.77%</td>
<td>-</td>
<td>-</td>
<td>27.92%</td>
<td>-</td>
<td>-</td>
<td>37.91%</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>I3D</td>
<td>78.46%</td>
<td>56.15%</td>
<td>77.5%</td>
<td>57.61%</td>
<td>48.98%</td>
<td>55.58%</td>
<td>54.02%</td>
<td>59.03%</td>
<td>61.67%</td>
</tr>
<tr>
<td>Audio+I3D</td>
<td>78.46%</td>
<td>55.96%</td>
<td>72.69%</td>
<td>57.61%</td>
<td>44.42%</td>
<td>58.38%</td>
<td>54.02%</td>
<td>46.42%</td>
<td>62.41%</td>
</tr>
</table>
