随机森林：
随机森林虽然理论上可以应对分类特征（非数据形式：字符串）和数据缺失，scikit-learn 实现却并不支持这两种情况。所以我们需要使用 pd.interpolate() 
来填充缺失的值，然后使用 pd.get_dummies() 的『One-Hot Encoding』来将分类特征转换为数字特征。
