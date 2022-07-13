# coordinate-descent-for-glm

A implement of optimization of glm via coordinate descent [[1]](#1).

### Usage

```
model = logisticRegression()
model.fit(X_train, y_train)
model.select('bic')
model.score(X_test, y_test)
```

<a id="1">[1]</a> 
Friedman, Jerome, Trevor Hastie, and Rob Tibshirani. "Regularization paths for generalized linear models via coordinate descent." Journal of statistical software 33.1 (2010): 1.
