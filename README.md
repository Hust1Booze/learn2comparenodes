# Learning to Compare Nodes in Branch and Bound with Graph Neural Networks

Abdel Ghani Labassi, Didier Chételat and Andrea Lodi

This is the official implementation of our [NeurIPS 2022 paper](https://arxiv.org/abs/2210.16934).

## Citation
Please cite our paper if you use this code in your work.
```
@inproceedings{conf/nips/labassi22,
  title={Learning to Compare Nodes in Branch and Bound with Graph Neural Networks},
  author={Abdel Ghani Labassi and Didier Chételat and Andrea Lodi},
  booktitle={Advances in Neural Information Processing Systems 35},
  year={2022}
}
```

## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.

优点：
1. 分支定界作为序列生成的过程,表征分支定界过程为序列
 模型可以看到历史的决策过程，以往的模型不行
 把 MILP 和BnB分开，每次分支定界不用encode MILP的信息，快

2. 网络结构
decoder（MILP） -decoder（BNB）结构，方式来融合MILP和BNB过程 的信息
3. 实验
对比算法 ： SCIP，
  选变量：Gauss2019，Retro, Yoshua Bengio(AAAI-21)
  选节点：L2C，INFORMS paper ...

  encode MILP 需要的时间
  BNB squence ATTENTION 热力图（可解释性）
  算法收敛速度（GAP）

问题 ？

绘图： 参考文章： Non-autoregressive Generative Models for Reranking Recommendation


问题：连续两次branch？