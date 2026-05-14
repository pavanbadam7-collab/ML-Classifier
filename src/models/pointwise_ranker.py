from sklearn.ensemble import RandomForestRegressor


def train_pointwise_model(X_train, y_train):
    """
    Train regression model to predict recovery amount.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def predict_recovery(model, X_test):
    """
    Predict recovered amount.
    """

    predictions = model.predict(X_test)

    return predictions