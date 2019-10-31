# Parallel Programming Project

## Playing Taiko by parallel reinforcement learning


## The participants
- 0856071 謝秉瑾
- 0856108 謝宗祐
- 0856160 洪鈺恆

## Introduction
In this project we will parallelly train multiple reinforcement learning agents to play taiko.

## Statement of the problem


Most reinforcement learning environment uses OpenAi atari gym environment. Because OpenAi gym had already parallelized well for each environment, all we have to do is parallelly collect data from each environment. However, in the real case, we have to parallelly control each environment and prevent data race and so on. It is more complex than using OpenAi atari gym environment.
## Proposed approaches

![](https://i.imgur.com/e7AIU5B.png)

## Language selection
- python threadind
- tensorflow
## Related work
[use deep q-learning playing atari](https://arxiv.org/abs/1312.5602)
[policy proximal optimization](https://arxiv.org/abs/1707.06347)
## Statement of expected results
We can parallelly train and play taiko.
## A timetable
10/29~11/13 Do research on taiko web and reinforcement methods.
11/13~11/20 Choose which reinforcement method to use and implement single thread version for this project.
11/20~11/27 Find which part of code can be parallelized.
11/27~12/10 Implement parallel training of taiko. 
12/10~ Train agents.
## References
https://github.com/bui/taiko-web
https://github.com/openai/baselines

## Requirements
- pip3 install -r requirements.txt

## TO DO
- [x] offline GAME
- [x] single thread running(4500 episode)
- [] multi-thread
- [] play on taiko-web
