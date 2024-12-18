import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchhydro_mini.configs.model_config import MODEL_PARAM_DICT
from torchhydro_mini.models.lstm import SimpleLSTM
from torchhydro_mini.models.kernel_conv import uh_conv, uh_gamma


class Hbv4Dpl(torch.nn.Module):
    def __init__(self, warmup_length, kernel_size=15):
        super(Hbv4Dpl, self).__init__()
        self.name = "HBV"
        self.params_names = MODEL_PARAM_DICT["hbv"]["param_name"]
        parasca_lst = MODEL_PARAM_DICT["hbv"]["param_range"]
        self.beta_scale = parasca_lst["BETA"]
        self.fc_scale = parasca_lst["FC"]
        self.k0_scale = parasca_lst["K0"]
        self.k1_scale = parasca_lst["K1"]
        self.k2_scale = parasca_lst["K2"]
        self.lp_scale = parasca_lst["LP"]
        self.perc_scale = parasca_lst["PERC"]
        self.uzl_scale = parasca_lst["UZL"]
        self.tt_scale = parasca_lst["TT"]
        self.cfmax_scale = parasca_lst["CFMAX"]
        self.cfr_scale = parasca_lst["CFR"]
        self.cwh_scale = parasca_lst["CWH"]
        self.a_scale = parasca_lst["A"]
        self.theta_scale = parasca_lst["THETA"]
        self.warmup_length = warmup_length
        self.kernel_size = kernel_size
        # there are 3 input vars in HBV: P, PET and TEMPERATURE
        self.feature_size = 3

    def forward(
        self, x, parameters, out_state=False, rout_opt=True
    ):
        hbv_device = x.device
        precision = 1e-5
        buffer_time = self.warmup_length
        # Initialization
        if buffer_time > 0:
            with torch.no_grad():
                x_init = x[0:buffer_time, :, :]
                warmup_length = 0
                init_model = Hbv4Dpl(warmup_length, kernel_size=self.kernel_size)
                if init_model.warmup_length > 0:
                    raise RuntimeError(
                        "Please set warmup_length as 0 when initializing HBV model"
                    )
                _, snowpack, meltwater, sm, suz, slz = init_model(
                    x_init, parameters, out_state=True, rout_opt=False
                )
        else:

            # Without buff time, initialize state variables with zeros
            n_grid = x.shape[1]
            snowpack = (torch.zeros(n_grid, dtype=torch.float32) + 0.001).to(hbv_device)
            meltwater = (torch.zeros(n_grid, dtype=torch.float32) + 0.001).to(
                hbv_device
            )
            sm = (torch.zeros(n_grid, dtype=torch.float32) + 0.001).to(hbv_device)
            suz = (torch.zeros(n_grid, dtype=torch.float32) + 0.001).to(hbv_device)
            slz = (torch.zeros(n_grid, dtype=torch.float32) + 0.001).to(hbv_device)

        # the sequence must be p, pet and t
        p_all = x[buffer_time:, :, 0]
        pet_all = x[buffer_time:, :, 1]
        t_all = x[buffer_time:, :, 2]

        # scale the parameters
        par_beta = self.beta_scale[0] + parameters[:, 0] * (
            self.beta_scale[1] - self.beta_scale[0]
        )
        # parCET = parameters[:,1]
        par_fc = self.fc_scale[0] + parameters[:, 1] * (
            self.fc_scale[1] - self.fc_scale[0]
        )
        par_k0 = self.k0_scale[0] + parameters[:, 2] * (
            self.k0_scale[1] - self.k0_scale[0]
        )
        par_k1 = self.k1_scale[0] + parameters[:, 3] * (
            self.k1_scale[1] - self.k1_scale[0]
        )
        par_k2 = self.k2_scale[0] + parameters[:, 4] * (
            self.k2_scale[1] - self.k2_scale[0]
        )
        par_lp = self.lp_scale[0] + parameters[:, 5] * (
            self.lp_scale[1] - self.lp_scale[0]
        )
        # parMAXBAS = parameters[:,7]
        par_perc = self.perc_scale[0] + parameters[:, 6] * (
            self.perc_scale[1] - self.perc_scale[0]
        )
        par_uzl = self.uzl_scale[0] + parameters[:, 7] * (
            self.uzl_scale[1] - self.uzl_scale[0]
        )
        # parPCORR = parameters[:,10]
        par_tt = self.tt_scale[0] + parameters[:, 8] * (
            self.tt_scale[1] - self.tt_scale[0]
        )
        par_cfmax = self.cfmax_scale[0] + parameters[:, 9] * (
            self.cfmax_scale[1] - self.cfmax_scale[0]
        )
        # parSFCF = parameters[:,13]
        par_cfr = self.cfr_scale[0] + parameters[:, 10] * (
            self.cfr_scale[1] - self.cfr_scale[0]
        )
        par_cwh = self.cwh_scale[0] + parameters[:, 11] * (
            self.cwh_scale[1] - self.cwh_scale[0]
        )

        n_step, n_grid = p_all.size()
        # Apply correction factor to precipitation
        # p_all = parPCORR.repeat(n_step, 1) * p_all

        # Initialize time series of model variables
        q_sim = (torch.zeros(p_all.size(), dtype=torch.float32) + 0.001).to(hbv_device)
        # # Debug for the state variables
        # SMlog = np.zeros(p_all.size())
        log_sm = np.zeros(p_all.size())
        log_ps = np.zeros(p_all.size())
        log_swet = np.zeros(p_all.size())
        log_re = np.zeros(p_all.size())

        for i in range(n_step):
            # Separate precipitation into liquid and solid components
            precip = p_all[i, :]
            tempre = t_all[i, :]
            potent = pet_all[i, :]
            rain = torch.mul(precip, (tempre >= par_tt).type(torch.float32))
            snow = torch.mul(precip, (tempre < par_tt).type(torch.float32))
            # snow = snow * parSFCF

            # Snow
            snowpack = snowpack + snow
            melt = par_cfmax * (tempre - par_tt)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > snowpack] = snowpack[melt > snowpack]
            melt = torch.min(melt, snowpack)
            meltwater = meltwater + melt
            snowpack = snowpack - melt
            refreezing = par_cfr * par_cfmax * (par_tt - tempre)
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > meltwater] = meltwater[refreezing > meltwater]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, meltwater)
            snowpack = snowpack + refreezing
            meltwater = meltwater - refreezing
            to_soil = meltwater - (par_cwh * snowpack)
            # to_soil[to_soil < 0.0] = 0.0
            to_soil = torch.clamp(to_soil, min=0.0)
            meltwater = meltwater - to_soil

            # Soil and evaporation
            soil_wetness = (sm / par_fc) ** par_beta
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (rain + to_soil) * soil_wetness

            # log for displaying
            log_sm[i, :] = sm.detach().cpu().numpy()
            log_ps[i, :] = (rain + to_soil).detach().cpu().numpy()
            log_swet[i, :] = (sm / par_fc).detach().cpu().numpy()
            log_re[i, :] = recharge.detach().cpu().numpy()

            sm = sm + rain + to_soil - recharge
            excess = sm - par_fc
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            sm = sm - excess
            evap_factor = sm / (par_lp * par_fc)
            # evap_factor[evap_factor < 0.0] = 0.0
            # evap_factor[evap_factor > 1.0] = 1.0
            evap_factor = torch.clamp(evap_factor, min=0.0, max=1.0)
            et_act = potent * evap_factor
            et_act = torch.min(sm, et_act)
            sm = torch.clamp(
                sm - et_act, min=precision
            )  # sm can not be zero for gradient tracking

            # Groundwater boxes
            suz = suz + recharge + excess
            perc = torch.min(suz, par_perc)
            suz = suz - perc
            q0 = par_k0 * torch.clamp(suz - par_uzl, min=0.0)
            suz = suz - q0
            q1 = par_k1 * suz
            suz = suz - q1
            slz = slz + perc
            q2 = par_k2 * slz
            slz = slz - q2
            q_sim[i, :] = q0 + q1 + q2

            # # for debug state variables
            # SMlog[t,:] = sm.detach().cpu().numpy()

        if rout_opt is True:  # routing
            temp_a = self.a_scale[0] + parameters[:, -2] * (
                self.a_scale[1] - self.a_scale[0]
            )
            temp_b = self.theta_scale[0] + parameters[:, -1] * (
                self.theta_scale[1] - self.theta_scale[0]
            )
            rout_a = temp_a.repeat(n_step, 1).unsqueeze(-1)
            rout_b = temp_b.repeat(n_step, 1).unsqueeze(-1)
            uh_from_gamma = uh_gamma(rout_a, rout_b, len_uh=self.kernel_size)
            rf = torch.unsqueeze(q_sim, -1)
            qs = uh_conv(rf, uh_from_gamma)

        else:
            qs = torch.unsqueeze(q_sim, -1)  # add a dimension
        return (qs, snowpack, meltwater, sm, suz, slz) if out_state is True else qs


class DplLstmHbv(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        kernel_size,
        warmup_length,
        param_limit_func="sigmoid",
        param_test_way="final",
    ):
        super(DplLstmHbv, self).__init__()
        self.dl_model = SimpleLSTM(
            input_size, output_size, hidden_size
        )
        self.pb_model = Hbv4Dpl(warmup_length, kernel_size)
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        return lstm_pbm(self.dl_model, self.pb_model, self.param_func, x, z)


def lstm_pbm(dl_model, pb_model, param_func, x, z):
    gen = dl_model(z)
    if torch.isnan(gen).any():
        raise ValueError("Error: NaN values detected. Check your data firstly!!!")
    # we set all params' values in [0, 1] and will scale them when forwarding
    if param_func == "sigmoid":
        params_ = F.sigmoid(gen)
    elif param_func == "clamp":
        params_ = torch.clamp(gen, min=0.0, max=1.0)
    else:
        raise NotImplementedError(
            "We don't provide this way to limit parameters' range!! Please choose sigmoid or clamp"
        )
    # just get one-period values, here we use the final period's values
    params = params_[-1, :, :]
    return pb_model(x[:, :, : pb_model.feature_size], params)
