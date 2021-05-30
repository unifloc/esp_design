import os
from multiprocessing import Pool
import pandas as pd
import well_model

import runner
import json

#params = dict(
#
#        gamma_gas=0.7,
#        gamma_oil=0.8,
#        gamma_wat=1,
#        rsb_m3m3=100,
#        rp_m3m3=100,
#        pb_atma=120,
#        t_res_C=90,
#        bob_m3m3=-1,
#        muob_cP=-1,
#        PVTcorr=0,
#        ksep_fr=0.6,
#
#        p_bhp_atm=70,
#
#        p_wh_atm=20,
#        t_wh_c=20,
#        h_list_m=2000,
#        h_pump_m=1800,
#        diam_list_mm_casing=150,
#        diam_list_mm_tube=73,
#
#        gas_fraction_intake_d=0.2,
#
#        qliq_sm3day=80,
#
#        n_dots_for_nodal=50,  # 50 оптимум для дебага, 30 для скорости, 20 маловато - прямые линии
#
#        qliq_sm3day_range=None,  # np.linspace(1, 150, 10),
#        fw_perc=20,
#        hydr_corr=1,
#        temp_method=2,
#
#        freq_Hz=53,
#
#        # pump_id = 1460,
#        pump_id=2753,
#
#        num_stages=200,
#
#        pi_sm3dayatm=0.7,
#
#        pres_atma=180,
#
#        calc_esp_new=1,
#        esp_head_m=1400,
#        ESP_gas_correct=5
#    )

params = dict(
        # 931 2БС10  1185
        gamma_gas=0.7,
        gamma_oil=0.844,
        gamma_wat=1.014,
        rsb_m3m3=85.54,
        rp_m3m3=85.54,
        pb_atma=123,
        t_res_C=86,
        bob_m3m3=1.166,
        muob_cP=1.01,
        PVTcorr=0,
        ksep_fr=0.8,

        p_bhp_atm=100,

        p_wh_atm=15,
        t_wh_c=20,
        h_list_m=2776,
        h_pump_m=2661,
        diam_list_mm_casing=159,
        diam_list_mm_tube=67.8,

        gas_fraction_intake_d=0.2,

        qliq_sm3day=80,

        n_dots_for_nodal=50,  # 50 оптимум для дебага, 30 для скорости, 20 маловато - прямые линии

        qliq_sm3day_range=80,  # np.linspace(1, 150, 10),
        fw_perc=22,
        hydr_corr=1,
        temp_method=2,

        freq_Hz=60,

        # pump_id = 1460,
        pump_id=1185,

        num_stages=200,

        pi_sm3dayatm=0.9,

        pres_atma=188,

        calc_esp_new=1,
        esp_head_m=2750,
        ESP_gas_correct=5  # поправка Кирилла
    )


if __name__ == '__main__':
    debug = -1
    num_simulations = 1000
    vba_version = '7.28'
    pumps_heads = [2600, 2800, 3000]
    params['n_dots_for_nodal'] = 15
    params['calc_esp_new'] = 1

    with open('distr.txt') as json_file:
        data = json.load(json_file)

    all_stats_q_new, pi_mc, dist_p_res = data['all_stats_q_new'], data['pi_mc'], data['dist_p_res']

    pools = [
        [params, [1185], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, 0],
        #[params, [748], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, 1],
        [params, [412], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, 2],
        #[params, [685], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, 3],
        [params, [649], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, 4],
        #[params, [553], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, 5],

        [params, [1111], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, 6],
        [params, [1258], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, 7]

    ]

    #pools = [[params, [1185], pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, i] for i in range(6)]
    # pools = pools[7:8]

    os.system("taskkill /f /im EXCEL.EXE")
    with Pool(len(pools)+1) as p:
        res = p.map(runner.esp_design_wrapper, pools)
    os.system("taskkill /f /im EXCEL.EXE")
