import os
import pandas as pd
import fnmatch
from hydromodel_dl.trainers.trainlogger import save_model_params_log
from hydrodatautils.foundation.hydro_statistic import calculate_and_record_metrics


def set_unit_to_var(ds):
    units_dict = ds.attrs["units"]
    for var_name, units in units_dict.items():
        if var_name in ds:
            ds[var_name].attrs["units"] = units
    if "units" in ds.attrs:
        del ds.attrs["units"]
    return ds


class Resulter:
    def __init__(self, cfgs) -> None:
        self.cfgs = cfgs
        self.result_dir = cfgs["data_cfgs"]["test_path"]
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    @property
    def pred_name(self):
        return f"epoch_{str(self.chosen_trained_epoch)}_flow_pred"

    @property
    def obs_name(self):
        return f"epoch_{str(self.chosen_trained_epoch)}_flow_obs"

    @property
    def chosen_trained_epoch(self):
        model_loader = self.cfgs["evaluation_cfgs"]["model_loader"]
        if model_loader["load_way"] == "specified":
            epoch_name = str(model_loader["test_epoch"])
        elif model_loader["load_way"] == "best":
            # NOTE: TO make it consistent with the name in case of model_loader["load_way"] == "pth", the name have to be "best"
            epoch_name = "best"
        elif model_loader["load_way"] == "latest":
            epoch_name = str(self.cfgs["training_cfgs"]["epochs"])
        elif model_loader["load_way"] == "pth":
            epoch_name = model_loader["pth_path"].split(os.sep)[-1]
        else:
            raise ValueError("Invalid load_way")
        return epoch_name

    def save_cfg(self, cfgs):
        # save the cfgs after training
        # update the cfgs with the latest one
        self.cfgs = cfgs
        param_file_exist = any(
            (
                fnmatch.fnmatch(file, "*.json")
                and "_stat" not in file  # statistics json file
                and "_dict" not in file  # data cache json file
            )
            for file in os.listdir(self.result_dir)
        )
        if not param_file_exist:
            # although we save params log during training, but sometimes we directly evaluate a model
            # so here we still save params log if param file does not exist
            # no param file was saved yet, here we save data and params setting
            save_model_params_log(cfgs, self.result_dir)

    def save_result(self, pred, obs):
        save_dir = self.result_dir
        flow_pred_file = os.path.join(save_dir, self.pred_name)
        flow_obs_file = os.path.join(save_dir, self.obs_name)
        pred = set_unit_to_var(pred)
        obs = set_unit_to_var(obs)
        pred.to_netcdf(flow_pred_file + ".nc")
        obs.to_netcdf(flow_obs_file + ".nc")

    def eval_result(self, preds_xr, obss_xr):
        target_col = self.cfgs["data_cfgs"]["target_cols"]
        evaluation_metrics = self.cfgs["evaluation_cfgs"]["metrics"]
        basin_ids = self.cfgs["data_cfgs"]["object_ids"]
        test_path = self.cfgs["data_cfgs"]["test_path"]
        fill_nan = self.cfgs["evaluation_cfgs"]["fill_nan"]
        #  Then evaluate the model metrics
        if self.cfgs["data_cfgs"]["dataset"] == "FloodEventDplDataset":
            target_col = target_col[:-1]
        if type(fill_nan) is list and len(fill_nan) != len(target_col):
            raise ValueError("length of fill_nan must be equal to target_col's")
        for i, col in enumerate(target_col):
            eval_log = {}
            obs = obss_xr[col].to_numpy()
            pred = preds_xr[col].to_numpy()
            eval_log = calculate_and_record_metrics(
                obs,
                pred,
                evaluation_metrics,
                col,
                fill_nan[i] if isinstance(fill_nan, list) else fill_nan,
                eval_log,
            )
            data = {}
            for metric, values in eval_log.items():
                clean_metric = metric.replace(f"of {col}", "").strip()
                data[clean_metric] = values
            df = pd.DataFrame(data, index=basin_ids)
            output_file = os.path.join(test_path, f"metric_{col}.csv")
            df.to_csv(output_file, index_label="basin_id")
