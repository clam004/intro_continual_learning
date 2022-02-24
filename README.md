# intro_continual_learning

This is a tutorial to connect the mathematics and machine learning theory to practical implementations addressing the continual learning problem of artificial intelligence. We will learn this in python by examining and decontructing a method called [elastic weight colsolidation](https://www.pnas.org/content/114/13/3521) (EWC).

There is a shortage of tutorials that aim to directly help the student connect the math to the code. This is especially true in artificial intelligence where so much is published everyday. While it is true that the average programmer can load a "out of the box" library in 10 minutes and be running the latest model for a common task in 15 minutes, this is very different from being able to understand the mathematical theory behind the advances, the practical implementation details needed for real world deployment and the ability to adapt and combine the fundamental concepts to new applications not yet imagined.

### What does elastic weight colsolidation do?

The ability to learn tasks in a sequential fashion is crucial to the development of artificial intelligence. When an artificila neural network is trained on a new training set, unless that new training set includes all the old tasks combined with the new task, it generally is subject to catastrophic forgetting, whereby learning to solve new task B accompanies degradation of performance at old task A. In contrast, human neural networks can maintain expertise on tasks that they have not experienced for a long time. EWC addresses this problem by selectively slowing down learning on the weights (ie parameters, synaptic strengths) important for those old tasks.

### Credit/References:

1. [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/content/114/13/3521)

2. [shivamsaboo17](https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks)
