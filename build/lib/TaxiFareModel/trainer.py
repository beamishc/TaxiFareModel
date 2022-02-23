# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from TaxiFareModel.encoders import CenterTransformer, DistanceTransformer,TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib
from sklearn.model_selection import cross_val_score
import xgboost as xgb


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[GB][London][beamishc]TaxiFareModel1.0"

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

        # create center distance pipeline
        center_pipe = Pipeline([
            ('center_trans', CenterTransformer()),
            ('stdscaler', StandardScaler())
        ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('center distance', center_pipe, ['dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # Add the model of your choice to the pipeline
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('model', xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3))
        ])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_param('estimator', 'xgb')
        self.mlflow_log_param('hyperparams', 'n_estimators=200, max_depth=3, learning_rate=0.1')
        self.mlflow_log_param('data', '<2000 fare')
        self.mlflow_log_param('new_feature', 'center_distance')
        self.mlflow_log_metric('rmse', rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """Save the trained model into a model.joblib file"""
        filename = 'model.joblib'
        joblib.dump(self.pipeline, filename)


if __name__ == "__main__":
    # get data
    df = get_data(100_000)
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
    print(f'{round(cross_val_score(trainer.pipeline, X_train, y_train, cv=5).mean(),2)*100}%')
    # evaluate
    rmse = trainer.evaluate(X_test,y_test)
    print(f'rmse : {rmse}')
    # get experiment id
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
    # save model
    trainer.save_model()
