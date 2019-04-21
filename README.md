# competition_repositry
数据挖掘比赛相关记录：<br/>

模型相关笔记：<br/>

随机森林：随机森林虽然理论上可以应对分类特征（非数据形式：字符串）和数据缺失，scikit-learn 实现却并不支持这两种情况。所以我们需要使用 pd.interpolate() 来填充缺失的值，然后使用 pd.get_dummies() 的『One-Hot Encoding』来将分类特征转换为数字特征<br/>
gbm，GBDT，xgboost调优方法参考<br/>
https://blog.csdn.net/anshiquanshu/article/details/78753728<br/>
https://blog.csdn.net/anshiquanshu/article/details/78913722<br/>
https://blog.csdn.net/han_xiaoyang/article/details/52665396<br/>


更新记录：<br/>
2018.10 更新全国高校绿色计算大赛数据挖掘赛题，客户预测代码，Randomforest集成<br/>
2019.1 更新天池大数据竞赛津南数字制造算法挑战赛baseline代码，包括EDA，feature process，xgboost，lgb，cv，stacking<br/>




一些比较有意思的论文<br/>:
[1]Wang S , Minku L L , Yao X . A learning framework for online class imbalance learning[C]// Computational Intelligence and Ensemble Learning (CIEL), 2013 IEEE Symposium on. IEEE, 2013.
[2]Partalas I , Tsoumakas G , Katakis I , et al. Ensemble pruning using reinforcement learning.[C]// Conference on Advances in Artificial Intelligence. DBLP, 2006
[3]Douzas G, Bacao F. Effective data generation for imbalanced learning using conditional generative adversarial networks[J]. Expert Systems with applications, 2018, 91: 464-471.
[4] Yang P, Yoo P D, Fernando J, et al. “Sample Subset Optimization Techniques for Imbalanced and Ensemble Learning Problems in
Bioinformatics Applications,” IEEE Transactions on Cybernetics, vol.44, no. 3, pp. 445-455. 2014.
[5]Mnih V , Kavukcuoglu K , Silver D , et al. Playing Atari with Deep Reinforcement Learning[J]. Computer Science, 2013.


