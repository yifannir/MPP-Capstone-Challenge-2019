# MPP-Capstone-Challenge-2019
DAT264x: Identifying Accents in Spectrograms of Speech, Use just simple CNN get 0.86 score and 13th in LB

## problem describe
this problem can be discribe as an simple image classifier project, for detail https://www.datasciencecapstone.org/competitions/16/identifying-accents-speech/

## model design

we use cnn model to get the feature, and use image enhance to reduce variance. 

Our innovation is in image enhancement and model integration voting.

In image enhancement model, because Spectrograms are meanningful, so we just can crop image in X(time) aixs, and can not do any resize operation.

In image vote output model, because the image loader had croped the image, and different image block can get different output. What's more, when just one predict of the image block, the acc is very low, but when the image croped 1000 times, and vote the fowllowing 1000 predict labels, the acc can be a hug advance. 

the frame as below

<center>
    <img src="https://github.com/yifannir/MPP-Capstone-Challenge-2019/blob/master/model.png" width="66%"/>
</center>




## evalute result

<center class="half">
    <img src="https://github.com/yifannir/MPP-Capstone-Challenge-2019/blob/master/score.png" width="66%"/>
</center>


