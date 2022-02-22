# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # Add the model of your choice to the pipeline
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns = 'fare_amount').copy()
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # train
    trainer = Trainer(X_train,y_train)
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test,y_test)
    print(f'rmse : {rmse}')
