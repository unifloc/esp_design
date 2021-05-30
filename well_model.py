import pandas as pd
from numba import jit
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import json
from shapely.geometry import LineString, Point
import tqdm
import os
import UniflocVBA.v7_25.python_api as python_api_7_25
import UniflocVBA.v7_28.python_api as python_api_7_28

api, api_new = None, None
VBA_VERSION = '7.28'


def plot_pump_curve(q_arr, h_esp_arr, power_esp_arr, efficiency_esp_arr, z, esp_name, f=50, fnom=50, q_work=None,
                    show=True, xlabel=None):
    q_arr = q_arr * f / fnom
    h_esp_arr = h_esp_arr * (f / fnom) ** 2
    power_esp_arr = power_esp_arr * (f / fnom) ** 2
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines['right'].set_position(("axes", 1.15))

    p1, = ax.plot(q_arr, h_esp_arr, "b-", marker='o', label="Напор, м")
    p2, = twin1.plot(q_arr, power_esp_arr, "r-", marker='o', label="Мощность, Вт")
    p3, = twin2.plot(q_arr, efficiency_esp_arr, "g-", marker='o', label="КПД, д.ед.")

    if q_work is not None:
        f_interrr = interpolate.interp1d(q_arr, h_esp_arr, kind='cubic')
        p4 = ax.axvline(x=q_work, label=f"Рабочий режим Q={round(q_work, 2)}", linewidth=5, markersize=15)
        # p4, = ax.plot([q_work], [f_interrr(q_work)], "k",  marker = 'o', label="Рабочая точка",  markersize=15)

    # ax.axvspan(esp_df['Левая граница'].values[0]*f/fnom, esp_df['Правая граница'].values[0]*f/fnom,
    #           alpha=0.2, color='green') TODO вытащить из БД

    if xlabel is None:
        ax.set_xlabel("Подача, м3/сут")
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Напор, м")
    twin1.set_ylabel("Мощность, Вт")
    twin2.set_ylabel("КПД, д.ед.")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    if q_work is not None:
        ax.legend(handles=[p1, p2, p3, p4], loc='lower center')
    else:
        ax.legend(handles=[p1, p2, p3], loc='lower center')

    ax.grid()

    ax.set_title(f"{esp_name}, ступеней = {z} шт. при частоте = {f} Гц")
    if show:
        plt.show()
    else:
        additional = [p1, p2, p3, ax]
        return ax, twin1, twin2, additional


def interp_df(df, xname, yname, x_val, kind='linear'):
    """
    'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
    'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic
    """
    f = interpolate.interp1d(df[xname].astype(float).values, df[yname].astype(float).values, kind=kind)
    return f(x_val)


def calc_num_stages(params, api=api, api_new=api_new, vba_version=VBA_VERSION):
    if vba_version == '7.25':
        num_stages = int(params['esp_head_m'] / api.ESP_head_m(
            qliq_m3day=api.ESP_optRate_m3day(pump_id=params['pump_id'], freq_Hz=50),
            num_stages=1,
            freq_Hz=50,
            pump_id=params['pump_id'],
            mu_cSt=1,
            c_calibr=1,
            ))
    else:
        num_stages = int(params['esp_head_m'] / api_new.ESP_head_m(
            qliq_m3day=api_new.ESP_optRate_m3day(freq_Hz=50,
                                                 pump_id=params['pump_id'],
                                                 mu_cSt=1,
                                                 calibr_rate=1),
            num_stages=1,
            freq_Hz=50,
            pump_id=params['pump_id'],
            mu_cSt=1,
            calibr_head=1,
            calibr_rate=1,
            calibr_power=1))
    return num_stages


@jit(nopython=True)  # , fastmath=True)
def calc_QliqVogel_m3Day(Pi, Pr, P_test,
                         Wc, pb):
    if Pr < pb:
        pb = Pr

    qb = Pi * (Pr - pb)
    if Wc > 100:
        Wc = 100
    if Wc < 0:
        Wc = 0

    if (Wc == 100) or (P_test >= pb):

        calc_QliqVogel_m3Day = Pi * (Pr - P_test)

    else:
        fw = Wc / 100
        fo = 1 - fw
        qo_max = qb + (Pi * pb) / 1.8

        p_wfg = fw * (Pr - qo_max / Pi)

        if P_test > p_wfg:
            a = 1 + (P_test - (fw * Pr)) / (0.125 * fo * pb)
            b = fw / (0.125 * fo * pb * Pi)
            c = (2 * a * b) + 80 / (qo_max - qb)
            d = (a ** 2) - (80 * qb / (qo_max - qb)) - 81
            if b == 0:
                calc_QliqVogel_m3Day = abs(d / c)
            else:
                calc_QliqVogel_m3Day = (-c + ((c * c - 4 * b * b * d) ** 0.5)) / (2 * b ** 2)

        else:

            CG = 0.001 * qo_max
            cd = fw * (CG / Pi) + \
                 fo * 0.125 * pb * (-1 + (1 + 80 * ((0.001 * qo_max) / (qo_max - qb))) ** 0.5)
            calc_QliqVogel_m3Day = (p_wfg - P_test) / (cd / CG) + qo_max

    return calc_QliqVogel_m3Day


def create_normal_dist(mu, sigma, amount, plot=0, name='dist', show_inside = False):
    # mu, sigma = 100, 5 # mean and standard deviation

    s = np.random.normal(mu, sigma, amount)



    if plot > 0:
        count, bins, ignored = plt.hist(s, 30, density=True)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                 linewidth=2, color='r')
        plt.title(name)
        if show_inside:
            plt.show()

    return s


debug = True


def calc_esp_old(d, str_PVT_tube, p_intake, t_intake):
    # расчет эцн - переделать - нужно по ступеням
    # определение свойств ГЖС через насос
    q_mix_pump = api.MF_q_mix_rc_m3day(qliq_sm3day=d['qliq_sm3day'],
                                       fw_perc=d['fw_perc'],
                                       p_atma=p_intake,
                                       t_C=t_intake,
                                       str_PVT=str_PVT_tube)

    rho_mix_pump = api.MF_rho_mix_kgm3(qliq_sm3day=d['qliq_sm3day'],
                                       fw_perc=d['fw_perc'],
                                       p_atma=p_intake,
                                       t_C=t_intake,
                                       str_PVT=str_PVT_tube)

    mu_mix_pump = api.MF_mu_mix_cP(qliq_sm3day=d['qliq_sm3day'],
                                   fw_perc=d['fw_perc'],
                                   p_atma=p_intake,
                                   t_C=t_intake,
                                   str_PVT=str_PVT_tube)

    mu_sst = mu_mix_pump / (rho_mix_pump / 1000)

    p_esp_dis = api.ESP_head_m(qliq_m3day=q_mix_pump, num_stages=d['num_stages'],
                               freq_Hz=d['freq_Hz'],
                               pump_id=d['pump_id'],
                               mu_cSt=mu_sst,
                               c_calibr=1) * rho_mix_pump * 9.81 / 10 ** 5 + p_intake

    eff = api.ESP_eff_fr(qliq_m3day=q_mix_pump, num_stages=d['num_stages'],
                         freq_Hz=d['freq_Hz'],
                         pump_id=d['pump_id'],
                         mu_cSt=mu_sst,
                         c_calibr=1)

    head_esp = api.ESP_head_m(qliq_m3day=q_mix_pump, num_stages=d['num_stages'],
                              freq_Hz=d['freq_Hz'],
                              pump_id=d['pump_id'],
                              mu_cSt=mu_sst,
                              c_calibr=1)

    power_esp = api.ESP_power_W(qliq_m3day=q_mix_pump, num_stages=d['num_stages'],
                                freq_Hz=d['freq_Hz'],
                                pump_id=d['pump_id'],
                                mu_cSt=mu_sst,
                                c_calibr=1) / 1000

    gas_fraction_intake = -1

    return p_esp_dis, gas_fraction_intake, eff, head_esp, power_esp


def calc_esp_new(d, str_PVT_tube, p_intake, t_intake, m_api=api):
    api = m_api
    r = api.ESP_p_atma(qliq_sm3day=d['qliq_sm3day'],
                       fw_perc=d['fw_perc'],
                       p_calc_atma=p_intake,
                       num_stages=d['num_stages'],
                       freq_Hz=d['freq_Hz'],
                       pump_id=d['pump_id'],
                       str_PVT=str_PVT_tube,
                       t_intake_C=t_intake,
                       t_dis_C=t_intake,  # TODO
                       calc_along_flow=1,
                       ESP_gas_correct=d['ESP_gas_correct'],
                       c_calibr=1,
                       dnum_stages_integrate=1,
                       out_curves_num_points=20,
                       num_value=0,
                       q_gas_sm3day=0
                       )
    r = pd.DataFrame(r)

    eff = r[5][0]
    head_esp = r[1][0]
    power_esp = r[8][22] / 10 ** 3
    p_esp_dis = r[0][0]
    t_dis = r[4][22]
    gas_fraction_intake = r[4][0]
    q_mix_pump_mean = r.iloc[3:, 6].mean()
    return p_esp_dis, gas_fraction_intake, eff, head_esp, power_esp, r, q_mix_pump_mean


def calc_esp_new_7_28(params, str_PVT_tube, p_intake, t_intake, m_api=api_new):
    api_new = m_api
    # флюид
    encoded_fluid = api_new.encode_PVT(gamma_gas=params['gamma_gas'],
                                       gamma_oil=params['gamma_oil'],
                                       gamma_wat=params['gamma_wat'],
                                       rsb_m3m3=params['rsb_m3m3'],
                                       pb_atma=params['pb_atma'],
                                       t_res_C=params['t_res_C'],
                                       bob_m3m3=params['bob_m3m3'],
                                       muob_cP=params['muob_cP'],
                                       PVT_corr_set=params['PVTcorr'])
    # поток вместе с флюидом
    encoded_feed = api_new.encode_feed(q_liq_sm3day=params['qliq_sm3day'],
                                       fw_perc=params['fw_perc'],
                                       rp_m3m3=params['rp_m3m3'],
                                       q_gas_free_sm3day=-1,
                                       fluid=encoded_fluid
                                       )

    # модификация флюида
    encoded_feed_mode = api_new.feed_mod_separate_gas(k_sep=params['ksep_fr'],
                                                      p_atma=p_intake,
                                                      t_C=t_intake,
                                                      feed=encoded_feed,
                                                      param=''
                                                      )
    if params['temp_method'] == 2:
        t_dis_C = -1
    else:
        t_dis_C = t_intake
    # настройки вывода ЭЦН
    param = json.dumps({'show_array': 1})
    # расчет ЭЦН по UniflocVBA 7.28
    r = api_new.ESP_p_atma(
        p_calc_atma=p_intake,
        t_intake_C=t_intake,
        t_dis_C=t_dis_C,
        feed=encoded_feed_mode,
        pump_id=params['pump_id'],
        num_stages=params['num_stages'],
        freq_Hz=params['freq_Hz'],
        calc_along_flow=True,
        calibr_head=1,
        calibr_rate=1,
        calibr_power=1,
        gas_correct_model=params['ESP_gas_correct'],
        gas_correct_stage_by_stage=1,
        param=param,
    )

    r = pd.DataFrame(r)

    eff = r[6][0]
    head_esp = r[5][0]
    power_esp = r[8][0] / 10 ** 3
    p_esp_dis = r[0][0]
    t_dis = r[4][0]
    gas_fraction_intake = r[4][3]
    q_mix_pump_mean = r.iloc[3:, 5].mean()

    return p_esp_dis, gas_fraction_intake, eff, head_esp, power_esp, r, q_mix_pump_mean


def plot_pump_curves(params, head_esp, power_esp, eff, p_intake, t_intake, str_PVT_tube, q_mix_pump_mean,
                     qliq_on_surface=True, vba_version=VBA_VERSION, api=api, api_new=api_new):
    to_curve_params = params.copy()
    lqliq = to_curve_params['qliq_sm3day_range']
    lp_esp_dis, lgas_fraction_intake, leff, lhead_esp, lpower_esp, lq_mix_pump_mean = [], [], [], [], [], []
    for i in lqliq:
        to_curve_params['qliq_sm3day'] = i

        if vba_version != '7.28':
            m_api = api
            ip_esp_dis, igas_fraction_intake, ieff, ihead_esp, ipower_esp, _, iq_mix_pump_mean = calc_esp_new(
                to_curve_params,
                str_PVT_tube, p_intake, t_intake,
                m_api=m_api)
        else:
            m_api = api_new
            ip_esp_dis, igas_fraction_intake, ieff, ihead_esp, ipower_esp, _, iq_mix_pump_mean = calc_esp_new_7_28(
                to_curve_params,
                str_PVT_tube, p_intake, t_intake,
                m_api=m_api)

        lp_esp_dis.append(ip_esp_dis)
        lgas_fraction_intake.append(igas_fraction_intake)
        leff.append(ieff)
        lhead_esp.append(ihead_esp)
        lpower_esp.append(ipower_esp)
        lq_mix_pump_mean.append(iq_mix_pump_mean)

    if qliq_on_surface:
        q_liq_for_regime = params['qliq_sm3day']
        xlabel = 'Дебит жидкости в поверхностных условиях, м3/сут'
    else:
        q_liq_for_regime = q_mix_pump_mean
        lqliq = lq_mix_pump_mean
        xlabel = 'Средний расход ГЖС через насос, м3/сут'

    ax, twin1, twin2, _ = plot_pump_curve(np.array(lqliq), np.array(lhead_esp), np.array(lpower_esp),
                                          np.array(leff), 1,
                                          m_api.ESP_name(pump_id=to_curve_params['pump_id']) +
                                          f" OPT Rate = {round(m_api.ESP_optRate_m3day(pump_id=to_curve_params['pump_id']), 2)}",
                                          # f=to_curve_params['freq_Hz'] ,  #меняет частоту
                                          f=50,
                                          show=False, xlabel=xlabel)

    # добавим точки, расчитанные по модели, на характеристику
    if head_esp is not None:
        ax.plot([q_liq_for_regime], [head_esp], 'bd', markersize=20, label='Рабочий режим')
        twin1.plot([q_liq_for_regime], [power_esp], 'rd', markersize=20, label='Рабочий режим')
        twin2.plot([q_liq_for_regime], [eff], 'gd', markersize=20, label='Рабочий режим')

    plt.show()


def plot_well_curves(p_dis, p_esp_dis, str_PVT_tube, p_intake, t_intake, eff, head_esp, power_esp, gas_fraction_intake,
                     casing_pipe, tube_pipe, params):
    cas_df = pd.DataFrame(casing_pipe)
    tube_df = pd.DataFrame(tube_pipe)

    index_t_c = cas_df.iloc[2][cas_df.iloc[2] == 't,C'].index[0]
    index_t_amb_c = cas_df.iloc[2][cas_df.iloc[2] == 't_amb, C'].index[0]
    index_p_atm = cas_df.iloc[2][cas_df.iloc[2] == 'p,atma'].index[0]
    index_h_m = cas_df.iloc[2][cas_df.iloc[2] == 'h,m'].index[0]

    fig = plt.Figure()
    plt.plot(cas_df.iloc[3:, index_p_atm].values, cas_df.iloc[3:, index_h_m].values, label='КРД в обсадной колонне')
    plt.plot(tube_df.iloc[3:, index_p_atm].values, tube_df.iloc[3:, index_h_m].values, label='КРД в НКТ')
    plt.plot([params['p_bhp_atm']], params['h_list_m'], 'o',
             label=f"Забойное давление {round(params['p_bhp_atm'], 2)} атм")
    plt.plot([params['pres_atma']], params['h_list_m'], 'ro',
             label=f"Пластовое давление {round(params['pres_atma'], 2)} атм")
    plt.plot([p_dis], params['h_pump_m'], 'o', label=f"Давление на выкиде ЭЦН по НКТ {round(p_dis, 2)}, атм")
    plt.plot([p_esp_dis], params['h_pump_m'], 'o', label=f"Давление на выкиде ЭЦН по ЭЦН {round(p_esp_dis, 2)}, атм")
    plt.plot([p_intake], params['h_pump_m'], 'o', label=f"Давление на приеме ЭЦН {round(p_intake, 2)}, атм")
    plt.plot([params['p_wh_atm']], [0], 'o', label=f"Давление буферное {round(params['p_wh_atm'], 2)}, атм")

    plt.legend()
    plt.grid()
    plt.title(f"Кривая распределения давления для режима Qж = {round(params['qliq_sm3day'], 2)} м3/сут")
    plt.xlabel('Давление, атм')
    plt.ylabel('Глубина, м')
    plt.gca().invert_yaxis()
    plt.show()

    cas_df = pd.DataFrame(casing_pipe)
    tube_df = pd.DataFrame(tube_pipe)

    fig = plt.Figure()
    plt.plot(cas_df.iloc[3:, index_t_c].values, cas_df.iloc[3:, index_h_m].values, 'o-', label='КРТ в обсадной колонне')
    plt.plot(cas_df.iloc[3:, index_t_amb_c].values, cas_df.iloc[3:, index_h_m].values, 'o-',
             label='КРТ окр. среды в обсадной колонне')

    plt.plot(tube_df.iloc[3:, index_t_c].values, tube_df.iloc[3:, index_h_m].values, 'o-', label='КРТ в НКТ')
    plt.plot(tube_df.iloc[3:, index_t_amb_c].values, tube_df.iloc[3:, index_h_m].values, 'o-',
             label='КРТ окр. среды в НКТ')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    plt.title(f"Кривая распределения температуры для режима Qж = {round(params['qliq_sm3day'], 2)} м3/сут")
    plt.xlabel('Температура, С')
    plt.ylabel('Глубина измеренная, м')
    plt.show()


def calc_model(d, m_api=api):
    api = m_api
    # расчет в обсадной колонне КРД - до приема - снизу-вверх
    str_PVT_casing = api.PVT_encode_string(gamma_gas=d['gamma_gas'],
                                           gamma_oil=d['gamma_oil'],
                                           gamma_wat=d['gamma_wat'],
                                           rsb_m3m3=d['rsb_m3m3'],
                                           rp_m3m3=d['rp_m3m3'],
                                           pb_atma=d['pb_atma'],
                                           t_res_C=d['t_res_C'],
                                           bob_m3m3=d['bob_m3m3'],
                                           muob_cP=d['muob_cP'],
                                           PVTcorr=d['PVTcorr'],
                                           ksep_fr=-1,
                                           p_ksep_atma=-1,
                                           t_ksep_C=-1,
                                           gas_only=False)

    casing_pipe = api.MF_p_pipeline_atma(p_calc_from_atma=d['p_bhp_atm'],
                                         t_calc_from_C=d['t_res_C'],
                                         # t_val_C=d['t_wh_c'],
                                         t_val_C=[[0, d['t_wh_c']], [d['h_list_m'], d['t_res_C']]],
                                         h_list_m=[d['h_pump_m'], d['h_list_m']],
                                         diam_list_mm=d['diam_list_mm_casing'],
                                         qliq_sm3day=d['qliq_sm3day'],
                                         fw_perc=d['fw_perc'],
                                         q_gas_sm3day=0,
                                         str_PVT=str_PVT_casing,
                                         calc_flow_direction=0,
                                         hydr_corr=d['hydr_corr'],
                                         temp_method=2,
                                         c_calibr=1,
                                         roughness_m=0.0001,
                                         out_curves=2,
                                         out_curves_num_points=20,
                                         num_value=0,
                                         znlf=False)

    casing_pipe_df = pd.DataFrame(casing_pipe)
    p_intake = casing_pipe[0][0]
    t_intake = casing_pipe[0][1]

    # расчет нкт сверху вниз - определение давления на выкиде насоса
    str_PVT_tube = api.PVT_encode_string(gamma_gas=d['gamma_gas'],
                                         gamma_oil=d['gamma_oil'],
                                         gamma_wat=d['gamma_wat'],
                                         rsb_m3m3=d['rsb_m3m3'],
                                         rp_m3m3=d['rp_m3m3'],
                                         pb_atma=d['pb_atma'],
                                         t_res_C=d['t_res_C'],
                                         bob_m3m3=d['bob_m3m3'],
                                         muob_cP=d['muob_cP'],
                                         PVTcorr=d['PVTcorr'],
                                         ksep_fr=d['ksep_fr'],
                                         p_ksep_atma=float(p_intake),
                                         t_ksep_C=float(t_intake),
                                         gas_only=False)

    tube_pipe = api.MF_p_pipeline_atma(p_calc_from_atma=d['p_wh_atm'],
                                       t_calc_from_C=d['t_wh_c'],
                                       t_val_C=t_intake,
                                       h_list_m=d['h_pump_m'],
                                       diam_list_mm=d['diam_list_mm_tube'],
                                       qliq_sm3day=d['qliq_sm3day'],
                                       fw_perc=d['fw_perc'],
                                       q_gas_sm3day=0,
                                       str_PVT=str_PVT_tube,
                                       calc_flow_direction=10,
                                       hydr_corr=d['hydr_corr'],
                                       temp_method=0,
                                       c_calibr=1,
                                       roughness_m=0.0001,
                                       out_curves=2,
                                       out_curves_num_points=20,
                                       num_value=0,
                                       znlf=False)

    df = pd.DataFrame(tube_pipe)
    p_dis = tube_pipe[0][0]
    t_dis = tube_pipe[0][1]

    if d['calc_esp_new'] != 1:
        # print('calc_esp_old')
        p_esp_dis, gas_fraction_intake, eff, head_esp, power_esp = calc_esp_old(d, str_PVT_tube, p_intake, t_intake)
    else:
        # print('calc_esp_new')
        p_esp_dis, gas_fraction_intake, eff, head_esp, power_esp, esp_df, q_mix_pump_mean = calc_esp_new(d,
                                                                                                         str_PVT_tube,
                                                                                                         p_intake,
                                                                                                         t_intake,
                                                                                                         m_api=api)

    return p_dis, p_esp_dis, str_PVT_tube, p_intake, t_intake, eff, head_esp, power_esp, gas_fraction_intake, \
           casing_pipe, tube_pipe, q_mix_pump_mean, esp_df


def calc_model_new_7_28(d, m_api=api_new):
    ########### ЭЦН
    # флюид
    api_new = m_api
    encoded_fluid = api_new.encode_PVT(gamma_gas=d['gamma_gas'],
                                       gamma_oil=d['gamma_oil'],
                                       gamma_wat=d['gamma_wat'],
                                       rsb_m3m3=d['rsb_m3m3'],
                                       pb_atma=d['pb_atma'],
                                       t_res_C=d['t_res_C'],
                                       bob_m3m3=d['bob_m3m3'],
                                       muob_cP=d['muob_cP'],
                                       PVT_corr_set=d['PVTcorr'])

    # поток вместе с флюидом
    encoded_feed = api_new.encode_feed(q_liq_sm3day=d['qliq_sm3day'],
                                       fw_perc=d['fw_perc'],
                                       rp_m3m3=d['rp_m3m3'],
                                       q_gas_free_sm3day=-1,
                                       fluid=encoded_fluid
                                       )

    # температура
    encoded_t_model = api_new.encode_t_model(t_model=d['temp_method'],
                                             t_list_C=[
                                                 [0, d['t_wh_c']],
                                                 [d['h_list_m'], d['t_res_C']]
                                             ],
                                             t_start_C=d['t_res_C'] ,
                                             t_end_C=d['t_res_C'] ,
                                             param=''
                                             )

    # конструкция
    encoded_pipe = api_new.encode_pipe(h_list_m=[d['h_pump_m'], d['h_list_m']],
                                       diam_list_mm=d['diam_list_mm_casing'],
                                       roughness_m=0.0001)

    # настройки вывода ЭЦН
    param = json.dumps({'show_array': 1})

    # обсадная колонна

    casing_pipe = api_new.MF_pipe_p_atma(
        p_calc_from_atma=d['p_bhp_atm'],
        t_calc_from_C=d['t_res_C'],
        construction=encoded_pipe,
        feed=encoded_feed,
        t_model=encoded_t_model,
        calc_along_coord=False,
        flow_along_coord=False,
        flow_correlation=d['hydr_corr'],
        calibr_grav=1,
        calibr_fric=1,
        param=param,  # TODO добавить в вывод gasfraction

    )

    casing_df = pd.DataFrame(casing_pipe)

    p_intake = casing_df[0][0]
    t_intake = casing_df[2][0]

    ########### НКТ
    # модификация флюида для НКТ
    encoded_feed_mode = api_new.feed_mod_separate_gas(k_sep=d['ksep_fr'],
                                                      p_atma=p_intake,
                                                      t_C=t_intake,
                                                      feed=encoded_feed,
                                                      param=''
                                                      )

    # расчет НКТ

    str_PVT_tube = None  # для универсальности - в 7.25 отдельная строка

    # ЭЦН
    p_esp_dis, gas_fraction_intake, eff, head_esp, power_esp, esp_df, \
    q_mix_pump_mean = calc_esp_new_7_28(d, str_PVT_tube, p_intake, t_intake, m_api=api_new)
    if d['temp_method'] == 2:
        t_end_C = esp_df[4][0]
    else:
        t_end_C = t_intake

    # температура
    encoded_t_model = api_new.encode_t_model(t_model=0,
                                             t_list_C=[[0, d['t_wh_c']],
                                                       [d['h_pump_m'], t_intake]
                                                       ],
                                             t_start_C=d['t_wh_c'],
                                             t_end_C=t_end_C,
                                             param=''
                                             )

    # конструкция
    encoded_pipe = api_new.encode_pipe(h_list_m=[0, d['h_pump_m']],
                                       diam_list_mm=d['diam_list_mm_tube'],
                                       roughness_m=0.001)

    tube_pipe = api_new.MF_pipe_p_atma(
        p_calc_from_atma=d['p_wh_atm'],
        t_calc_from_C=d['t_wh_c'],
        construction=encoded_pipe,
        feed=encoded_feed_mode,
        t_model=encoded_t_model,
        calc_along_coord=True,
        flow_along_coord=False,
        flow_correlation=d['hydr_corr'],
        calibr_grav=1,
        calibr_fric=1,
        param=param,  # TODO добавить в вывод gasfraction
    )

    df = pd.DataFrame(tube_pipe)
    p_dis = tube_pipe[0][0]
    t_dis = tube_pipe[0][4]



    return p_dis, p_esp_dis, str_PVT_tube, p_intake, t_intake, eff, head_esp, power_esp, gas_fraction_intake, \
           casing_pipe, tube_pipe, q_mix_pump_mean, esp_df


def calc_all(d, debug=True, vba_version=VBA_VERSION, api=api, api_new=api_new):
    print('\n')

    if vba_version == '7.28':
        print('UniflocVBA 7.28')
        f_ipr_qliq_sm3day = api_new.IPR_q_liq_sm3day
        f_ipr_pwf_atma = api_new.IPR_p_wf_atma
        f_calc_model = calc_model_new_7_28
        m_api = api_new

    else:
        print('UniflocVBA 7.25')
        f_ipr_qliq_sm3day = api.IPR_qliq_sm3day
        f_ipr_pwf_atma = api.IPR_pwf_atma
        f_calc_model = calc_model
        m_api = api

    q_max = f_ipr_qliq_sm3day(d['pi_sm3dayatm'],
                              d['pres_atma'],
                              0,
                              d['fw_perc'],
                              d['pb_atma']) - 1

    d['qliq_sm3day_range'] = list(np.linspace(1, q_max * 0.2, int(d['n_dots_for_nodal'] / 3))) + \
                             list(np.linspace(q_max * 0.21, q_max * 0.7, int(d['n_dots_for_nodal']))) + \
                             list(np.linspace(q_max * 0.71, q_max, int(d['n_dots_for_nodal'] * 1.2)))
    # d['qliq_sm3day_range'] = np.logspace(q_max, 1, d['n_dots_for_nodal'])

    # создание пустых массивов для узлового анализа на выкиде насоса
    p_dis_pipe, p_dis_pump, p_intake_list, p_bhp_by_ipr_list = [], [], [], []

    # прогон по дебиту - расчет модели с разными дебитами - для решения узлового анализа - поиска рабочего режима
    pump_in_flowing_mode_count = 0
    for j, i in enumerate(d['qliq_sm3day_range']):
        #print(i)
        d['qliq_sm3day'] = i

        # расчет забойного давления для данного дебита
        d['p_bhp_atm'] = f_ipr_pwf_atma(
            d['pi_sm3dayatm'],
            d['pres_atma'],
            d['qliq_sm3day'],
            d['fw_perc'],
            d['pb_atma'],
        )

        p_dis, p_esp_dis, str_PVT_tube, p_intake, t_intake, eff, head_esp, power_esp, gas_fraction_intake, \
        casing_pipe, tube_pipe, q_mix_pump_mean, esp_df = f_calc_model(d, m_api=m_api)
        if debug == 2:
            plot_well_curves(p_dis, p_esp_dis, str_PVT_tube, p_intake, t_intake, eff, head_esp, power_esp, \
                             gas_fraction_intake, casing_pipe, tube_pipe, d)

        p_dis_pipe.append(p_dis)
        p_dis_pump.append(p_esp_dis)
        p_intake_list.append(p_intake)
        p_bhp_by_ipr_list.append(d['p_bhp_atm'])

        if p_esp_dis < p_intake:
            pump_in_flowing_mode_count += 1
            if pump_in_flowing_mode_count == 2 and j != (len(d['qliq_sm3day_range']) - 1):
                d['qliq_sm3day_range'] = d['qliq_sm3day_range'][:j + 1]
                break

    #### определение режима работы скважины с помощью узлового анализа

    # чистка данных от None
    df = pd.DataFrame({'qliq_sm3day_range': d['qliq_sm3day_range'],
                       'p_dis_pipe': p_dis_pipe,
                       'p_dis_pump': p_dis_pump,
                       'p_intake': p_intake_list,
                       'p_bhp_by_ipr_list': p_bhp_by_ipr_list

                       })
    df = df.dropna()

    d['qliq_sm3day_range'] = df['qliq_sm3day_range'].values.flatten()
    p_dis_pipe = df['p_dis_pipe'].values.flatten()
    p_dis_pump = df['p_dis_pump'].values.flatten()
    p_intake_list = df['p_intake'].values.flatten()
    p_bhp_by_ipr_list = df['p_bhp_by_ipr_list'].values.flatten()

    if len(p_dis_pump) <= 1:
        print('error - all none')
        return None, None, None, None, None, None, None, 0, None, None, None, None

    # построение графика узлового анализа
    if debug >= 1:
        fig = plt.Figure()
        plt.plot(d['qliq_sm3day_range'], p_dis_pipe, 'o-', label='Давление на выкиде ЭЦН, атм (НКТ)')
        plt.plot(d['qliq_sm3day_range'], p_dis_pump, 'o-', label='Давление на выкиде ЭЦН, атм (ЭЦН)')
        plt.plot(d['qliq_sm3day_range'], p_intake_list, 'o-', label='Давление на приеме ЭЦН, атм (ЭЦН)')
        plt.plot(d['qliq_sm3day_range'], p_bhp_by_ipr_list, 'o-', label='Давление забойное, атм')
        plt.plot(d['qliq_sm3day_range'], d['qliq_sm3day_range'] * 0 + d['pres_atma'], '--',
                 label='Пластовое давление, атм')
        plt.plot(d['qliq_sm3day_range'], d['qliq_sm3day_range'] * 0 + d['pb_atma'], '--',
                 label='Давление насыщения, атм')
        plt.plot(d['qliq_sm3day_range'], d['qliq_sm3day_range'] * 0 + d['p_wh_atm'], '--',
                 label='Давление устьевое, атм')

    # решение узлового анализа - нахождение режимного дебита
    first_line = LineString(np.column_stack((d['qliq_sm3day_range'], np.array(p_dis_pipe))))
    second_line = LineString(np.column_stack((d['qliq_sm3day_range'], np.array(p_dis_pump))))
    intersection = first_line.intersection(second_line)

    if intersection.geom_type == 'Point':
        d['qliq_sm3day'], p_solve = float(intersection.x), float(intersection.y)
        q_liq = d['qliq_sm3day']
    elif intersection.geom_type == 'MultiPoint':
        d['qliq_sm3day'], p_solve = float(intersection.bounds[-2]), float(intersection.bounds[-1])
        q_liq = d['qliq_sm3day']
    else:
        print('\n no solution\n')
        head_esp, z, eff, power_esp = None, None, None, None
        return None, None, d['h_pump_m'], head_esp, z, eff, power_esp, 0, None, None, None, None, None, None, None
    if debug >= 1:
        plt.plot([d['qliq_sm3day']], [p_solve], 'ro', markersize=12,
                 label=f"Режимное значение дебита = {round(d['qliq_sm3day'], 2)} м3/сут")
        plt.title('Узловой анализ на выкиде ЭЦН')
        plt.ylim(0, max([d['pres_atma']] + list(p_dis_pump)) + 10)
        plt.legend()
        plt.grid()
        plt.ylabel('Давление, МПа')
        plt.xlabel('Дебит жидкости, м3/сут')
        plt.show()

    ################# фактический расчет текущего режима с получением искомых параметров

    # определение фактического значения дебита
    d['p_bhp_atm'] = f_ipr_pwf_atma(
        d['pi_sm3dayatm'],
        d['pres_atma'],
        d['qliq_sm3day'],
        d['fw_perc'],
        d['pb_atma'],
    )

    p_dis, p_esp_dis, str_PVT_tube, p_intake, t_intake, eff, head_esp, power_esp, gas_fraction_intake, \
    casing_pipe, tube_pipe, q_mix_pump_mean, esp_df = f_calc_model(d, m_api=m_api)
    if debug >= 1:
        plot_well_curves(p_dis, p_esp_dis, str_PVT_tube, p_intake, t_intake, eff, head_esp, power_esp, \
                         gas_fraction_intake, casing_pipe, tube_pipe, d)
        plot_pump_curves(d, head_esp, power_esp, eff, p_intake, t_intake, str_PVT_tube, q_mix_pump_mean,
                         qliq_on_surface=False, vba_version=vba_version, api=api, api_new=api_new)
        plot_pump_curves(d, head_esp, power_esp, eff, p_intake, t_intake, str_PVT_tube, q_mix_pump_mean,
                         qliq_on_surface=True, vba_version=vba_version, api=api, api_new=api_new)

    print('norm calculation sucessful')
    p_bhp_atm = d['p_bhp_atm']

    json_params = json.dumps({key: value for (key, value) in d.items() if type(value) is not np.ndarray})

    return casing_pipe, tube_pipe, d['h_pump_m'], head_esp, eff, \
           power_esp, q_liq, 1, gas_fraction_intake, p_bhp_atm, p_dis, p_esp_dis, json_params, q_mix_pump_mean, esp_df

def save_in_df(lh_mes, lhead_esp, leff, lpower_esp, lqliq, lstatus, lgas_fraction_intake,
                   lp_bhp_atm, lp_esp_dis, lparams, num_simulations, lp_esp_dis_by_tube,
              lq_mix_pump_mean,  lesp_df,  lcasing_pipe, ltube_pipe):

    df = pd.DataFrame([lh_mes, lhead_esp, leff, lpower_esp, lqliq, lstatus, lgas_fraction_intake,
                       lp_bhp_atm, lp_esp_dis, lparams, range(num_simulations), lp_esp_dis_by_tube,
                      lq_mix_pump_mean,  lesp_df,  lcasing_pipe, ltube_pipe]).T.dropna()


    df.columns =  ['lh_mes', 'lhead_esp', 'leff', 'lpower_esp',
                 'lqliq', 'lstatus', 'lgas_fraction_intake', 'lp_bhp_atm',
                 'lp_esp_dis', 'lparams', 'simnumber', 'lp_esp_dis_by_tube', 'lq_mix_pump_mean',  'lesp_df',
                   'lcasing_pipe', 'ltube_pipe']

    df = df.sort_values(by = 'leff')
    return df


def create_q_dist(params, pi_mean=0.8, pi_std=0.15,
                  pres_std=10,
                  num_simulations=1_000_00, debug=debug):
    all_stats_q_new = []
    if debug >0:
        fig = plt.Figure()
    pi_mc = create_normal_dist(pi_mean, pi_std, num_simulations, name='Кпрод', plot=debug)
    if debug > 0:
        plt.xlabel('Коэффициент продуктивности, м3/сут/атм')
        plt.ylabel('Плотность вероятности')
        plt.title('Распределение коэффициента продуктивности')
        plt.show()
    if debug > 0:
        fig = plt.Figure()
    dist_p_res = create_normal_dist(params['pres_atma'], pres_std, num_simulations, name='Pres', plot=debug)
    if debug > 0:
        plt.xlabel('Пластовое давление, атм')
        plt.ylabel('Плотность вероятности')
        plt.title('Распределение пластового давления')
        plt.show()

    for i in range(num_simulations):
        new_q = calc_QliqVogel_m3Day(Pi=np.random.choice(pi_mc),
                                     P_test=params['p_bhp_atm'],
                                     Pr=np.random.choice(dist_p_res),
                                     Wc=params['fw_perc'],
                                     pb=params['pb_atma']
                                     )
        all_stats_q_new.append(new_q)

    if debug > 0:
        fig = plt.Figure()

        fig, ax = plt.subplots()

        count, bins, ignored = plt.hist(all_stats_q_new, 300, [0, 400], density=True,
                                        label=f"Qж ср = {round(np.mean(all_stats_q_new), 3)} м3/сут")

        plt.axvline(x=np.quantile(all_stats_q_new, q=0.5), c='r')

        plt.title('Распределение дебита жидкости, м3/сут')
        plt.xlabel('Дебит жидкости, м3/сут')
        plt.ylabel('Плотность вероятности')
        ax.legend()
        plt.show()
    return all_stats_q_new, pi_mc, dist_p_res


def run_design(params, pumps_ids, pumps_heads, pi_mc, dist_p_res, debug=1, num_simulations=1, api=api, api_new=api_new,
               vba_version=VBA_VERSION, thread_number = 0, path=r'C:\Git\probability_calculations' + '\\'):
    os.chdir(path + "calc_new")
    results = []
    for this_pump_id in pumps_ids:
        params['pump_id'] = this_pump_id
        print(f"pump_id = {this_pump_id}")
        for this_pump_head in pumps_heads:
            params['esp_head_m'] = this_pump_head
            print(f"esp_head_m = {this_pump_head}")

            params['num_stages'] = calc_num_stages(params, api=api, api_new=api_new, vba_version=vba_version)

            lh_mes, lhead_esp, leff, lpower_esp, lqliq, \
            lstatus, lgas_fraction_intake, lp_bhp_atm, lp_esp_dis, lparams, lp_esp_dis_by_tube, \
            lq_mix_pump_mean, lesp_df, \
            lcasing_pipe, ltube_pipe = [], [], [], \
                                       [], [], [], [], [], [], [], [], [], [], [], []

            for j, i in tqdm.tqdm(enumerate(range(num_simulations))):
                # for j, i in enumerate(range(num_simulations)):

                print('\n')
                print(f"iter = {j}")
                this_pi = np.random.choice(pi_mc)
                params['pi_sm3dayatm'] = np.random.choice(pi_mc)
                params['pres_atma'] = np.random.choice(dist_p_res)
                if this_pi < 0:
                    this_pi = 1

                casing_pipe, tube_pipe, h_mes, head_esp, eff, power_esp, q_liq, status, gas_fraction_intake, p_bhp_atm, \
                p_dis, p_esp_dis, json_params, q_mix_pump_mean, esp_df = calc_all(params, debug=debug,
                                                                                  vba_version=vba_version,
                                                                                  api=api, api_new=api_new)

                cas_df = pd.DataFrame(casing_pipe)
                tube_df = pd.DataFrame(tube_pipe)

                lh_mes.append(h_mes)
                lhead_esp.append(head_esp)
                leff.append(eff)
                lpower_esp.append(power_esp)
                lqliq.append(q_liq)
                lstatus.append(status)
                lgas_fraction_intake.append(gas_fraction_intake)
                lp_bhp_atm.append(p_bhp_atm)
                lp_esp_dis.append(p_esp_dis)
                lparams.append(json_params)
                lp_esp_dis_by_tube.append(p_dis)
                lq_mix_pump_mean.append(q_mix_pump_mean)
                if esp_df is not None:
                    lesp_df.append(esp_df.to_json())
                    lcasing_pipe.append(json.dumps(casing_pipe))
                    ltube_pipe.append(json.dumps(tube_pipe))
                else:
                    lesp_df.append(esp_df)
                    lcasing_pipe.append(casing_pipe)
                    ltube_pipe.append(tube_pipe)

            df = save_in_df(lh_mes, lhead_esp, leff, lpower_esp, lqliq, lstatus, lgas_fraction_intake,
                            lp_bhp_atm, lp_esp_dis, lparams, num_simulations, lp_esp_dis_by_tube,
                            lq_mix_pump_mean, lesp_df, lcasing_pipe, ltube_pipe
                            )
            df.to_csv(f"res_pump_thread_{thread_number}_id_{this_pump_id}_head_{this_pump_head}.csv")
            results.append((this_pump_id, this_pump_head, df.copy()))

    os.chdir(path)
    return results


if __name__ == '__main__':

    api = python_api_7_25.API("UniflocVBA/v7_25/UniflocVBA_7.xlam")


    api_new = python_api_7_28.API("UniflocVBA/v7_28/UniflocVBA_7_28.xlam")
    api_new.encode_PVT()

    params = dict(

        gamma_gas=0.7,
        gamma_oil=0.8,
        gamma_wat=1,
        rsb_m3m3=100,
        rp_m3m3=100,
        pb_atma=120,
        t_res_C=90,
        bob_m3m3=-1,
        muob_cP=-1,
        PVTcorr=0,
        ksep_fr=0.6,

        p_bhp_atm=70,

        p_wh_atm=20,
        t_wh_c=20,
        h_list_m=2000,
        h_pump_m=1800,
        diam_list_mm_casing=150,
        diam_list_mm_tube=73,

        gas_fraction_intake_d=0.2,

        qliq_sm3day=80,

        n_dots_for_nodal=50,  # 50 оптимум для дебага, 30 для скорости, 20 маловато - прямые линии

        qliq_sm3day_range=None,  # np.linspace(1, 150, 10),
        fw_perc=20,
        hydr_corr=1,
        temp_method=2,

        freq_Hz=53,

        # pump_id = 1460,
        pump_id=2753,

        num_stages=200,

        pi_sm3dayatm=0.7,

        pres_atma=180,

        calc_esp_new=1,
        esp_head_m=1400,
        ESP_gas_correct=20
    )

    debug = 0

    all_stats_q_new, pi_mc, dist_p_res = create_q_dist(params, pi_mean = 0.8, pi_std = 0.15,
                  pres_std = 10,
                  num_simulations= 1_000_00, debug=debug)

    # для одного расчета
    pumps_ids = [1153,  # 80
                 2753,  # 100 проблемная

                 ]
    pumps_heads = [1500]

    num_simulations = 1
    params['n_dots_for_nodal'] = 15
    params['calc_esp_new'] = 1

    results = run_design(params, pumps_ids, pumps_heads, pi_mc, dist_p_res, debug=debug, num_simulations=1, api=None,
                         api_new=api_new,
                         vba_version='7.28')

