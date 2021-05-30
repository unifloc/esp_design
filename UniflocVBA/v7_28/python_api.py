H_CORRELATION = 0 # 0 - BeggsBrill, 1 - Ansari and so on 
PVT_CORRELATION = 0 # 0 -Standing, 1 -McCain, 2 - linear 
PVT_DEFAULT = "gamma_gas:0,900;gamma_oil:0,750;gamma_wat:1,000;rsb_m3m3:100,000;rp_m3m3:-1,000;pb_atma:-1,000;tres_C:90,000;bob_m3m3:-1,000;muob_cP:-1,000;PVTcorr:0;ksep_fr:0,000;pksep_atma:-1,000;tksep_C:-1,000; " 
ESP_DEFAULT = "ESP_ID:1006.00000;HeadNom_m:2000.00000;ESPfreq_Hz:50.00000;ESP_U_V:1000.00000;MotorPowerNom_kW:30.00000;Tintake_C:85.00000;t_dis_C:25.00000;KsepGS_fr:0.00000;ESP_energy_fact_Whday:0.00000;ESP_cable_type:0;ESP_Hmes_m:0.00000;ESP_gas_degradation_type:0;c_calibr_head:0.00000;PKV_work_min:-1,00000;PKV_stop_min:-1,00000;"
WELL_DEFAULT = "hperf_m:2000,00000;hpump_m:1800,00000;udl_m:0,00000;d_cas_mm:150,00000;dtub_mm:72,00000;dchoke_mm:15,00000;roughness_m:0,00010;tbh_C:85,00000;twh_C:25,00000;"
WELL_GL_DEFAULT = "hperf_m:2500,00000;htub_m:2000,00000;udl_m:0,00000;d_cas_mm:125,00000;dtub_mm:62,00000;dchoke_mm:15,00000;roughness_m:0,00010;tbh_C:100,00000;twh_C:50,00000;GLV:1;H_glv_m:1500,000;d_glv_mm:5,000;p_glv_atma:50,000;"
const_gg_ = 0.6 
const_gw_ = 1 
const_go_ = 0.86 
const_sigma_wat_gas_Nm = 0.01 
const_sigma_oil_Nm = 0.025 
const_mu_w = 0.36
const_mu_g = 0.0122 
const_mu_o = 0.7 
const_rsb_default = 100 
const_Bob_default = 1.2 
const_tres_default = 90 
const_Roughness_default = 0.0001 
StartEndTemp = 0 
Standing_based = 0 
const_rho_air = 1.2217 
 
import xlwings as xw
addin_name_str = "UniflocVBA_7.xlam"
class API():
    def __init__(self, addin_name_str):
        self.book = xw.Book(addin_name_str)
    def MF_dpdl_atmm(self, d_m,p_atma,Ql_rc_m3day,Qg_rc_m3day,mu_oil_cP=const_mu_o,mu_gas_cP=const_mu_g,sigma_oil_gas_Nm=const_sigma_oil_Nm,rho_lrc_kgm3=const_go_*1000,rho_grc_kgm3=const_gg_*const_rho_air,eps_m=0.0001,theta_deg=90,hcorr=1,param_out=0,calibr_grav=1,calibr_fric=1):
        """
 ========== description ============== 
расчет градиента давления с использованием многофазных корреляций 
        
 ==========  arguments  ============== 

     d_m - диаметр трубы в которой идет поток    

     p_atma - давление в точке расчета    

     ql_rc_m3day - дебит жидкости в рабочих условиях    

     qg_rc_m3day - дебит газа в рабочих условиях    

     mu_oil_cp - вязкость нефти в рабочих условиях    

     mu_gas_cp - вязкость газа в рабочих условиях    

     sigma_oil_gas_nm - поверхностное натяжение  жидкость газ    

     rho_lrc_kgm3 - плотность нефти    

     rho_grc_kgm3 - плотность газа    

     eps_m - шероховатость    

     theta_deg - угол от горизонтали    

     hcorr - тип корреляции    

     param_out - параметр для вывода    

     calibr_grav - калибровка гравитации    

     calibr_fric - калибровка трения   

        """

        self.f_MF_dpdl_atmm = self.book.macro("MF_dpdl_atmm")
        return self.f_MF_dpdl_atmm(d_m,p_atma,Ql_rc_m3day,Qg_rc_m3day,mu_oil_cP,mu_gas_cP,sigma_oil_gas_Nm,rho_lrc_kgm3,rho_grc_kgm3,eps_m,theta_deg,hcorr,param_out,calibr_grav,calibr_fric)

    def MF_choke_calibr(self, feed,d_choke_mm,p_in_atma=-1,p_out_atma=-1,d_pipe_mm=70,t_choke_C=20,param=""):
        """
 ========== description ============== 
 расчет корректирующего фактора (множителя) модели штуцера под замеры  медленный расчет - калибровка подбирается 
        
 ==========  arguments  ============== 

     feed - закодированная строка с параметрами потока.    

     d_choke_mm - диаметр штуцера (эффективный), мм    

     p_in_atma - давление на входе (высокой стороне)    

     p_out_atma - давление на выходе (низкой стороне)    

     d_pipe_mm - диаметр трубы до и после штуцера, мм    

     t_choke_c - температура, с.    

     param - параметры расчета json строка   

        """

        self.f_MF_choke_calibr = self.book.macro("MF_choke_calibr")
        return self.f_MF_choke_calibr(feed,d_choke_mm,p_in_atma,p_out_atma,d_pipe_mm,t_choke_C,param)

    def MF_pipe_p_atma(self, p_calc_from_atma,t_calc_from_C,construction="",feed="",t_model="",calc_along_coord=True,flow_along_coord=True,flow_correlation=0,calibr_grav=1,calibr_fric=1,param=""):
        """
 ========== description ============== 
 расчет распределения давления и температуры в трубопроводе  выводит результат в виде таблицы значений 
        
 ==========  arguments  ============== 

     p_calc_from_atma - давление с которого начинается расчет, атм  граничное значение для проведения расчета    

     t_calc_from_c - температура в точке где задано давление расчета    

     construction - параметры конструкции json строка. используйте  функцию encode_pipe() для генерации    

     feed - параметры потока флюидов json строка. используйте  функцию encode_feed() для генерации  construction - параметры конструкции json строка. используйте  функцию encode_pi..см.мануал   

     t_model - параметры температурной модели json строка.  используйте функцию encode_feed() для генерации    

     calc_along_coord - направление расчета относительно координат.    

     flow_along_coord - направление потока относительно координат.  flow_correl ation - гидравлическая корреляция, номер    

   flow_correlation   

     calibr_grav - калибровка на гравитационную составляющую  градиента давления    

     calibr_fric - калибровка на составляющую трения  градиента давления    

     param - дополнительные параметры расчета потока   

        """

        self.f_MF_pipe_p_atma = self.book.macro("MF_pipe_p_atma")
        return self.f_MF_pipe_p_atma(p_calc_from_atma,t_calc_from_C,construction,feed,t_model,calc_along_coord,flow_along_coord,flow_correlation,calibr_grav,calibr_fric,param)

    def MF_pipe_calc(self, p_calc_from_atma,t_calc_from_C,construction="",feed="",t_model="",calc_along_coord=True,flow_along_coord=True,flow_correlation=0,calibr_grav=1,calibr_fric=1,param=""):
        """
 ========== description ============== 
 расчет распределения давления и температуры в трубопроводе  выводит полный набор результатов в виде массива json параметров 
        
 ==========  arguments  ============== 

     p_calc_from_atma - давление с которого начинается расчет, атм  граничное значение для проведения расчета    

     t_calc_from_c - температура в точке где задано давление расчета    

     construction - параметры конструкции json строка. используйте  функцию encode_pipe() для генерации    

     feed - параметры потока флюидов json строка. используйте  функцию encode_feed() для генерации  construction - параметры конструкции json строка. используйте  функцию encode_pi..см.мануал   

     t_model - параметры температурной модели json строка.  используйте функцию encode_feed() для генерации    

     calc_along_coord - направление расчета относительно координат.    

     flow_along_coord - направление потока относительно координат.  flow_correl ation - гидравлическая корреляция, номер    

   flow_correlation   

     calibr_grav - калибровка на гравитационную составляющую  градиента давления    

     calibr_fric - калибровка на составляющую трения  градиента давления    

     param - дополнительные параметры расчета потока   

        """

        self.f_MF_pipe_calc = self.book.macro("MF_pipe_calc")
        return self.f_MF_pipe_calc(p_calc_from_atma,t_calc_from_C,construction,feed,t_model,calc_along_coord,flow_along_coord,flow_correlation,calibr_grav,calibr_fric,param)

    def MF_choke_q_sm3day(self, feed,d_choke_mm,p_in_atma,p_out_atma,t_choke_C=20,d_pipe_mm=70,calibr=1,param=""):
        """
 ========== description ============== 
 расчет давления в штуцере 
        
 ==========  arguments  ============== 

     feed - закодированная строка с параметрами потока.    

     d_choke_mm - диаметр штуцера (эффективный)    

     p_in_atma - давление на входе в штуцер, атм.  высокая сторона    

     p_out_atma - давление на выходе из штуцера, атм.  низкая сторона    

     t_choke_c - температура потока, с.    

     d_pipe_mm - диаметр трубы до и после штуцера    

   calibr   

     param - параметры расчета json строка   

        """

        self.f_MF_choke_q_sm3day = self.book.macro("MF_choke_q_sm3day")
        return self.f_MF_choke_q_sm3day(feed,d_choke_mm,p_in_atma,p_out_atma,t_choke_C,d_pipe_mm,calibr,param)

    def MF_choke_calc(self, d_choke_mm,feed,p_in_atma=-1,p_out_atma=-1,t_choke_C=20,d_pipe_mm=70,calibr=1,param=""):
        """
 ========== description ============== 
 расчет штуцера (дросселя) с выводом всех параметров расчета  тип расчета зависит от исходных данных 
        
 ==========  arguments  ============== 

     d_choke_mm - диаметр штуцера (эффективный)    

     feed - закодированная строка с параметрами потока.    

     p_in_atma - давление на входе в штуцер, атм  если задано, то используется в расчете    

     p_out_atma - давление на выходе из штуцера, атм  если задано, то используется в расчете    

     t_choke_c - температура потока, с.    

     d_pipe_mm - диаметр трубы до и после штуцера    

     calibr - базовая калибровка штуцера (на расход)    

     param - параметры расчета json строка   

        """

        self.f_MF_choke_calc = self.book.macro("MF_choke_calc")
        return self.f_MF_choke_calc(d_choke_mm,feed,p_in_atma,p_out_atma,t_choke_C,d_pipe_mm,calibr,param)

    def MF_choke_p_atma(self, d_choke_mm,feed,p_calc_from_atma,t_choke_C=20,d_pipe_mm=70,calc_along_flow=True,calibr=1,param=""):
        """
 ========== description ============== 
 расчет давления в штуцере (дросселе) 
        
 ==========  arguments  ============== 

     d_choke_mm - диаметр штуцера (эффективный)    

     feed - закодированная строка с параметрами потока.    

     p_calc_from_atma - давление с которого начинается расчет, атм  граничное значение для проведения расчета  либо давление на входе, либо на выходе    

     t_choke_c - температура потока, с.    

     d_pipe_mm - диаметр трубы до и после штуцера    

     calc_along_flow - флаг направления расчета относительно потока    

   calibr   

     param - параметры расчета json строка   

        """

        self.f_MF_choke_p_atma = self.book.macro("MF_choke_p_atma")
        return self.f_MF_choke_p_atma(d_choke_mm,feed,p_calc_from_atma,t_choke_C,d_pipe_mm,calc_along_flow,calibr,param)

    def MF_choke_calibr_fast(self, feed,d_choke_mm,p_in_atma=-1,p_out_atma=-1,d_pipe_mm=70,t_choke_C=20,param=""):
        """
 ========== description ============== 
 расчет корректирующего фактора (множителя) модели штуцера под замеры  быстрый расчет - калибровка вычисляется 
        
 ==========  arguments  ============== 

     feed - закодированная строка с параметрами потока.    

     d_choke_mm - диаметр штуцера (эффективный), мм    

     p_in_atma - давление на входе (высокой стороне)    

     p_out_atma - давление на выходе (низкой стороне)    

     d_pipe_mm - диаметр трубы до и после штуцера, мм    

     t_choke_c - температура, с.    

     param - параметры расчета json строка   

        """

        self.f_MF_choke_calibr_fast = self.book.macro("MF_choke_calibr_fast")
        return self.f_MF_choke_calibr_fast(feed,d_choke_mm,p_in_atma,p_out_atma,d_pipe_mm,t_choke_C,param)

    def PVT_calc(self, p_atma,t_C,PVT_prop,param=""):
        """
 ========== description ============== 
 функция расчета всех PVT свойств нефти при заданных  давлении и температуре 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - статичные свойства флюида - плотности rsb и пр  используйте encode_pvt для генерации    

     param - набор параметров расчета в виде json строки   

        """

        self.f_PVT_calc = self.book.macro("PVT_calc")
        return self.f_PVT_calc(p_atma,t_C,PVT_prop,param)

    def PVT_bg_m3m3(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 функция расчета объемного коэффициента газа 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_bg_m3m3 = self.book.macro("PVT_bg_m3m3")
        return self.f_PVT_bg_m3m3(p_atma,t_C,PVT_prop)

    def PVT_bo_m3m3(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет объемного коэффициента нефти 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_bo_m3m3 = self.book.macro("PVT_bo_m3m3")
        return self.f_PVT_bo_m3m3(p_atma,t_C,PVT_prop)

    def PVT_bw_m3m3(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет объемного коэффициента воды 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_bw_m3m3 = self.book.macro("PVT_bw_m3m3")
        return self.f_PVT_bw_m3m3(p_atma,t_C,PVT_prop)

    def PVT_salinity_ppm(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет солености воды 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_salinity_ppm = self.book.macro("PVT_salinity_ppm")
        return self.f_PVT_salinity_ppm(p_atma,t_C,PVT_prop)

    def PVT_mu_oil_cP(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет вязкости нефти 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_mu_oil_cP = self.book.macro("PVT_mu_oil_cP")
        return self.f_PVT_mu_oil_cP(p_atma,t_C,PVT_prop)

    def PVT_mu_gas_cP(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет вязкости газа 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_mu_gas_cP = self.book.macro("PVT_mu_gas_cP")
        return self.f_PVT_mu_gas_cP(p_atma,t_C,PVT_prop)

    def PVT_mu_wat_cP(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет вязкости воды 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_mu_wat_cP = self.book.macro("PVT_mu_wat_cP")
        return self.f_PVT_mu_wat_cP(p_atma,t_C,PVT_prop)

    def PVT_rs_m3m3(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет газосодержания 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_rs_m3m3 = self.book.macro("PVT_rs_m3m3")
        return self.f_PVT_rs_m3m3(p_atma,t_C,PVT_prop)

    def PVT_z(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет коэффициента сверхсжимаемости газа 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_z = self.book.macro("PVT_z")
        return self.f_PVT_z(p_atma,t_C,PVT_prop)

    def PVT_rho_oil_kgm3(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет плотности нефти в рабочих условиях 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_rho_oil_kgm3 = self.book.macro("PVT_rho_oil_kgm3")
        return self.f_PVT_rho_oil_kgm3(p_atma,t_C,PVT_prop)

    def PVT_rho_gas_kgm3(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет плотности газа в рабочих условиях 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_rho_gas_kgm3 = self.book.macro("PVT_rho_gas_kgm3")
        return self.f_PVT_rho_gas_kgm3(p_atma,t_C,PVT_prop)

    def PVT_rho_wat_kgm3(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет плотности воды в рабочих условиях 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_rho_wat_kgm3 = self.book.macro("PVT_rho_wat_kgm3")
        return self.f_PVT_rho_wat_kgm3(p_atma,t_C,PVT_prop)

    def PVT_pb_atma(self, t_C,PVT_prop):
        """
 ========== description ============== 
 Расчет давления насыщения 
        
 ==========  arguments  ============== 

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_pb_atma = self.book.macro("PVT_pb_atma")
        return self.f_PVT_pb_atma(t_C,PVT_prop)

    def PVT_ST_oilgas_Nm(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет коэффициента поверхностного натяжения нефть - газ 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_ST_oilgas_Nm = self.book.macro("PVT_ST_oilgas_Nm")
        return self.f_PVT_ST_oilgas_Nm(p_atma,t_C,PVT_prop)

    def PVT_ST_watgas_Nm(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет коэффициента поверхностного натяжения вода - газ 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_ST_watgas_Nm = self.book.macro("PVT_ST_watgas_Nm")
        return self.f_PVT_ST_watgas_Nm(p_atma,t_C,PVT_prop)

    def PVT_ST_liqgas_Nm(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет коэффициента поверхностного натяжения жидкость - газ 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_ST_liqgas_Nm = self.book.macro("PVT_ST_liqgas_Nm")
        return self.f_PVT_ST_liqgas_Nm(p_atma,t_C,PVT_prop)

    def PVT_cp_oil_JkgC(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет теплоемкости нефти при постоянном давлении cp 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_cp_oil_JkgC = self.book.macro("PVT_cp_oil_JkgC")
        return self.f_PVT_cp_oil_JkgC(p_atma,t_C,PVT_prop)

    def PVT_cp_gas_JkgC(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет теплоемкости газа при постоянном давлении cp 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_cp_gas_JkgC = self.book.macro("PVT_cp_gas_JkgC")
        return self.f_PVT_cp_gas_JkgC(p_atma,t_C,PVT_prop)

    def PVT_cv_gas_JkgC(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет теплоемкости газа при постоянном давлении cp 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_cv_gas_JkgC = self.book.macro("PVT_cv_gas_JkgC")
        return self.f_PVT_cv_gas_JkgC(p_atma,t_C,PVT_prop)

    def PVT_cp_wat_JkgC(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет теплоемкости воды при постоянном давлении cp 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_cp_wat_JkgC = self.book.macro("PVT_cp_wat_JkgC")
        return self.f_PVT_cp_wat_JkgC(p_atma,t_C,PVT_prop)

    def PVT_compressibility_wat_1atm(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет сжимаемости воды 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_compressibility_wat_1atm = self.book.macro("PVT_compressibility_wat_1atm")
        return self.f_PVT_compressibility_wat_1atm(p_atma,t_C,PVT_prop)

    def PVT_compressibility_oil_1atm(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет сжимаемости нефти 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_compressibility_oil_1atm = self.book.macro("PVT_compressibility_oil_1atm")
        return self.f_PVT_compressibility_oil_1atm(p_atma,t_C,PVT_prop)

    def PVT_compressibility_gas_1atm(self, p_atma,t_C,PVT_prop):
        """
 ========== description ============== 
 расчет сжимаемости нефти 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     pvt_prop - строка с параметрами флюида,  используйте encode_pvt для ее генерации   

        """

        self.f_PVT_compressibility_gas_1atm = self.book.macro("PVT_compressibility_gas_1atm")
        return self.f_PVT_compressibility_gas_1atm(p_atma,t_C,PVT_prop)

    def feed_calc(self, p_atma,t_C,feed,param=""):
        """
 ========== description ============== 
 функция расчета параметров потока 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации    

     param - параметры расчета и вывода результатов   

        """

        self.f_feed_calc = self.book.macro("feed_calc")
        return self.f_feed_calc(p_atma,t_C,feed,param)

    def feed_gas_fraction_d(self, p_atma,t_C,feed,ksep_add_fr=0):
        """
 ========== description ============== 
 функция расчета коэффициента Джоуля Томсона 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации    

     ksep_add_fr - коэффициент сепарации газа из потока   

        """

        self.f_feed_gas_fraction_d = self.book.macro("feed_gas_fraction_d")
        return self.f_feed_gas_fraction_d(p_atma,t_C,feed,ksep_add_fr)

    def feed_p_gas_fraction_atma(self, free_gas_d,t_C,feed,ksep_add_fr=0):
        """
 ========== description ============== 
 расчет давления при котором  достигается заданная доля газа в потоке 
        
 ==========  arguments  ============== 

     free_gas_d - допустимая доля газа в потоке;    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации    

     ksep_add_fr - коэффициент сепарации газа из потока   

        """

        self.f_feed_p_gas_fraction_atma = self.book.macro("feed_p_gas_fraction_atma")
        return self.f_feed_p_gas_fraction_atma(free_gas_d,t_C,feed,ksep_add_fr)

    def feed_rp_gas_fraction_m3m3(self, free_gas_d,p_atma,t_C,feed,ksep_add_fr=0):
        """
 ========== description ============== 
 расчет газового фактора  при котором достигается заданная доля газа в потоке 
        
 ==========  arguments  ============== 

     free_gas_d - допустимая доля газа в потоке;    

   p_atma   

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации    

     ksep_add_fr - коэффициент сепарации газа из потока   

        """

        self.f_feed_rp_gas_fraction_m3m3 = self.book.macro("feed_rp_gas_fraction_m3m3")
        return self.f_feed_rp_gas_fraction_m3m3(free_gas_d,p_atma,t_C,feed,ksep_add_fr)

    def feed_cJT_Katm(self, p_atma,t_C,feed):
        """
 ========== description ============== 
 функция расчета коэффициента Джоуля Томсона 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации   

        """

        self.f_feed_cJT_Katm = self.book.macro("feed_cJT_Katm")
        return self.f_feed_cJT_Katm(p_atma,t_C,feed)

    def feed_q_mix_rc_m3day(self, p_atma,t_C,feed):
        """
 ========== description ============== 
 функция расчета расхода газо жидкостной смеси (ГЖС) 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации   

        """

        self.f_feed_q_mix_rc_m3day = self.book.macro("feed_q_mix_rc_m3day")
        return self.f_feed_q_mix_rc_m3day(p_atma,t_C,feed)

    def feed_rho_mix_kgm3(self, p_atma,t_C,feed):
        """
 ========== description ============== 
 функция расчета плотности газо жидкостной смеси (ГЖС) 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации   

        """

        self.f_feed_rho_mix_kgm3 = self.book.macro("feed_rho_mix_kgm3")
        return self.f_feed_rho_mix_kgm3(p_atma,t_C,feed)

    def feed_mu_mix_cP(self, p_atma,t_C,feed):
        """
 ========== description ============== 
 функция расчета плотности газо жидкостной смеси (ГЖС) 
        
 ==========  arguments  ============== 

     p_atma - давление, атм    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации   

        """

        self.f_feed_mu_mix_cP = self.book.macro("feed_mu_mix_cP")
        return self.f_feed_mu_mix_cP(p_atma,t_C,feed)

    def feed_mod_separate_gas(self, k_sep,p_atma,t_C,feed,param=""):
        """
 ========== description ============== 
 функция расчета свойст потока после сепарации газа 
        
 ==========  arguments  ============== 

     k_sep - коэффициент сепарации газа    

     p_atma - давление, атм    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации    

     param - параметры расчета и вывода результатов   

        """

        self.f_feed_mod_separate_gas = self.book.macro("feed_mod_separate_gas")
        return self.f_feed_mod_separate_gas(k_sep,p_atma,t_C,feed,param)

    def feed_mod_split(self, k_sep_gas,k_sep_oil,k_sep_wat,p_atma,t_C,feed,param=""):
        """
 ========== description ============== 
 функция расчета свойств разделенного потока флюидов 
        
 ==========  arguments  ============== 

     k_sep_gas - коэффициент сепарации газа    

     k_sep_oil - коэффициент сепарации газа    

     k_sep_wat - коэффициент сепарации газа    

     p_atma - давление, атм    

     t_c - температура, с.    

     feed - параметры потока флюидов, дебит, обводненность и пр  используйте encode_feed для генерации    

     param - параметры расчета и вывода результатов   

        """

        self.f_feed_mod_split = self.book.macro("feed_mod_split")
        return self.f_feed_mod_split(k_sep_gas,k_sep_oil,k_sep_wat,p_atma,t_C,feed,param)

    def feed_mod_mix(self, feed_1,feed_2,param=""):
        """
 ========== description ============== 
 функция расчета свойств разделенного потока флюидов 
        
 ==========  arguments  ============== 

   feed_1   

   feed_2   

     param - параметры расчета и вывода результатов   

        """

        self.f_feed_mod_mix = self.book.macro("feed_mod_mix")
        return self.f_feed_mod_mix(feed_1,feed_2,param)

    def IPR_q_liq_sm3day(self, pi_sm3dayatm,p_res_atma,p_wf_atma,fw_perc=0,pb_atma=-1):
        """
 ========== description ============== 
 расчет дебита по давлению и продуктивности 
        
 ==========  arguments  ============== 

     pi_sm3dayatm - коэффициент продуктивности, ст.м3/сут/атм    

     p_res_atma - пластовое давление, абс. атм    

     p_wf_atma - забойное давление, абс. атм    

     fw_perc - обводненность, %    

     pb_atma - давление насыщения, абс. атм   

        """

        self.f_IPR_q_liq_sm3day = self.book.macro("IPR_q_liq_sm3day")
        return self.f_IPR_q_liq_sm3day(pi_sm3dayatm,p_res_atma,p_wf_atma,fw_perc,pb_atma)

    def IPR_p_wf_atma(self, pi_sm3dayatm,p_res_atma,q_liq_sm3day,fw_perc=0,pb_atma=-1):
        """
 ========== description ============== 
 расчет забойного давления по дебиту и продуктивности 
        
 ==========  arguments  ============== 

     pi_sm3dayatm - коэффициент продуктивности, ст.м3/сут/атм    

     p_res_atma - пластовое давление, абс. атм    

     q_liq_sm3day - дебит жидкости скважины на поверхности, ст.м3/сут    

     fw_perc - обводненность, %    

     pb_atma - давление насыщения, абс. атм   

        """

        self.f_IPR_p_wf_atma = self.book.macro("IPR_p_wf_atma")
        return self.f_IPR_p_wf_atma(pi_sm3dayatm,p_res_atma,q_liq_sm3day,fw_perc,pb_atma)

    def IPR_pi_sm3dayatm(self, Qtest_sm3day,pwf_test_atma,p_res_atma,fw_perc=0,pb_atma=-1):
        """
 ========== description ============== 
 расчет коэффициента продуктивности пласта  по данным тестовой эксплуатации 
        
 ==========  arguments  ============== 

     qtest_sm3day - тестовый дебит скважины, ст.м3/сут    

     pwf_test_atma - тестовое забойное давление, абс. атм    

     p_res_atma - пластовое давление, абс. атм    

     fw_perc - обводненность, %    

     pb_atma - давление насыщения, абс. атм   

        """

        self.f_IPR_pi_sm3dayatm = self.book.macro("IPR_pi_sm3dayatm")
        return self.f_IPR_pi_sm3dayatm(Qtest_sm3day,pwf_test_atma,p_res_atma,fw_perc,pb_atma)

    def ESP_head_m(self, qliq_m3day,num_stages=1,freq_Hz=50,pump_id=737,mu_cSt=-1,calibr_head=1,calibr_rate=1,calibr_power=1):
        """
 ========== description ============== 
 номинальный напор ЭЦН (на основе каталога ЭЦН)  учитывается поправка на вязкость и калибровки 
        
 ==========  arguments  ============== 

     qliq_m3day - дебит жидкости в условиях насоса (стенд)    

     num_stages - количество ступеней    

     freq_hz - частота вращения насоса    

     pump_id - номер насоса в базе данных    

     mu_cst - вязкость жидкости, сст;    

     calibr_head - калибровка (множитель) на напор    

     calibr_rate - калибровка (множитель) на расход    

     calibr_power - калибровка (множитель) на мощность   

        """

        self.f_ESP_head_m = self.book.macro("ESP_head_m")
        return self.f_ESP_head_m(qliq_m3day,num_stages,freq_Hz,pump_id,mu_cSt,calibr_head,calibr_rate,calibr_power)

    def ESP_power_W(self, qliq_m3day,num_stages=1,freq_Hz=50,pump_id=737,mu_cSt=-1,calibr_head=1,calibr_rate=1,calibr_power=1):
        """
 ========== description ============== 
 номинальная мощность потребляемая ЭЦН с вала (на основе каталога ЭЦН)  учитывается поправка на вязкость 
        
 ==========  arguments  ============== 

     qliq_m3day - дебит жидкости в условиях насоса (стенд)    

     num_stages - количество ступеней    

     freq_hz - частота вращения насоса    

     pump_id - номер насоса в базе данных    

     mu_cst - вязкость жидкости, сст;    

     calibr_head - калибровка (множитель) на напор    

     calibr_rate - калибровка (множитель) на расход    

     calibr_power - калибровка (множитель) на мощность   

        """

        self.f_ESP_power_W = self.book.macro("ESP_power_W")
        return self.f_ESP_power_W(qliq_m3day,num_stages,freq_Hz,pump_id,mu_cSt,calibr_head,calibr_rate,calibr_power)

    def ESP_eff_fr(self, qliq_m3day,num_stages=1,freq_Hz=50,pump_id=737,mu_cSt=-1,calibr_head=1,calibr_rate=1,calibr_power=1):
        """
 ========== description ============== 
 номинальный КПД ЭЦН (на основе каталога ЭЦН)  учитывается поправка на вязкость 
        
 ==========  arguments  ============== 

     qliq_m3day - дебит жидкости в условиях насоса (стенд)    

     num_stages - количество ступеней    

     freq_hz - частота вращения насоса    

     pump_id - номер насоса в базе данных    

     mu_cst - вязкость жидкости, сст;    

     calibr_head - калибровка (множитель) на напор    

     calibr_rate - калибровка (множитель) на расход    

     calibr_power - калибровка (множитель) на мощность   

        """

        self.f_ESP_eff_fr = self.book.macro("ESP_eff_fr")
        return self.f_ESP_eff_fr(qliq_m3day,num_stages,freq_Hz,pump_id,mu_cSt,calibr_head,calibr_rate,calibr_power)

    def ESP_name(self, pump_id):
        """
 ========== description ============== 
 название ЭЦН по номеру 
        
 ==========  arguments  ============== 

     pump_id - идентификатор насоса в базе данных   

        """

        self.f_ESP_name = self.book.macro("ESP_name")
        return self.f_ESP_name(pump_id)

    def ESP_rate_max_sm3day(self, freq_Hz=50,pump_id=737,mu_cSt=-1,calibr_rate=1):
        """
 ========== description ============== 
 максимальный дебит ЭЦН для заданной частоты  по номинальной кривой РНХ 
        
 ==========  arguments  ============== 

     freq_hz - частота вращения эцн    

     pump_id - идентификатор насоса в базе данных    

     mu_cst - вязкость для расчета поправок    

     calibr_rate - калибровка на расход   

        """

        self.f_ESP_rate_max_sm3day = self.book.macro("ESP_rate_max_sm3day")
        return self.f_ESP_rate_max_sm3day(freq_Hz,pump_id,mu_cSt,calibr_rate)

    def ESP_optRate_m3day(self, freq_Hz=50,pump_id=737,mu_cSt=-1,calibr_rate=1):
        """
 ========== description ============== 
 оптимальный дебит ЭЦН для заданной частоты  по номинальной кривой РНХ 
        
 ==========  arguments  ============== 

     freq_hz - частота вращения эцн    

     pump_id - идентификатор насоса в базе данных    

   mu_cst   

   calibr_rate  

        """

        self.f_ESP_optRate_m3day = self.book.macro("ESP_optRate_m3day")
        return self.f_ESP_optRate_m3day(freq_Hz,pump_id,mu_cSt,calibr_rate)

    def ESP_id_by_rate(self, q):
        """
 ========== description ============== 
 функция возвращает идентификатор типового насоса по значению  номинального дебита 
        
 ==========  arguments  ============== 

     q - номинальный дебит   

        """

        self.f_ESP_id_by_rate = self.book.macro("ESP_id_by_rate")
        return self.f_ESP_id_by_rate(q)

    def ESP_p_atma(self, p_calc_atma,t_intake_C=50,t_dis_C=50,feed="",pump_id=737,num_stages=1,freq_Hz=50,calc_along_flow=True,calibr_head=1,calibr_rate=1,calibr_power=1,gas_correct_model=0,gas_correct_stage_by_stage=0,param=""):
        """
 ========== description ============== 
функция расчета давления на выходе/входе ЭЦН в рабочих условиях большинство параметров задается явно 
        
 ==========  arguments  ============== 

     p_calc_atma - давление для которого делается расчет  либо давление на приеме насоса  либо давление на выкиде насоса    

     t_intake_c - температура на приеме насоcа    

     t_dis_c - температура на выкиде насоса.    

     feed - параметры потока флюидов json строка. используйте  функцию encode_feed() для генерации    

     pump_id - идентификатор насоса    

     num_stages - количество ступеней    

     freq_hz - частота вращения вала эцн, гц    

     определяется параметром calc_along_flow  t_intake_c - температура на приеме насоcа  t_dis_c - температура на выкиде насоса.  если = 0 и calc_along_flow = 1 то рассчитывается ..см.мануал   

     calibr_head - калибровка (множитель) на напор    

     calibr_rate - калибровка (множитель) на расход    

     calibr_power - калибровка (множитель) на мощность    

     gas_correct_model - модель калибровки по газу    

     gas_correct_stage_by_stage - модель применятеся  для всех ступеней по отдельности    

     param - дополнительные параметры расчета потока   

        """

        self.f_ESP_p_atma = self.book.macro("ESP_p_atma")
        return self.f_ESP_p_atma(p_calc_atma,t_intake_C,t_dis_C,feed,pump_id,num_stages,freq_Hz,calc_along_flow,calibr_head,calibr_rate,calibr_power,gas_correct_model,gas_correct_stage_by_stage,param)

    def GLV_q_gas_sm3day(self, d_mm,p_in_atma,p_out_atma,gamma_g,t_C,calibr=1):
        """
 ========== description ============== 
 функция расчета расхода газа через газлифтный клапан/штуцер  результат массив значений и подписей 
        
 ==========  arguments  ============== 

     d_mm - диаметр основного порта клапана, мм    

     p_in_atma - давление на входе в клапан (затруб), атма    

     p_out_atma - давление на выходе клапана (нкт), атма    

     gamma_g - удельная плотность газа    

     t_c - температура клапана, с    

   calibr  

        """

        self.f_GLV_q_gas_sm3day = self.book.macro("GLV_q_gas_sm3day")
        return self.f_GLV_q_gas_sm3day(d_mm,p_in_atma,p_out_atma,gamma_g,t_C,calibr)

    def GLV_p_atma(self, d_mm,p_calc_atma,q_gas_sm3day,gamma_g=0.6,t_C=25,calc_along_flow=False,p_open_atma=0,calibr=1):
        """
 ========== description ============== 
 функция расчета давления на входе или на выходе  газлифтного клапана (простого) при закачке газа.  результат массив значений и подписей 
        
 ==========  arguments  ============== 

     d_mm - диаметр клапана, мм    

     p_calc_atma - давление на входе (выходе) клапана, атма    

     q_gas_sm3day - расход газа, ст. м3/сут    

     gamma_g - удельная плотность газа    

     t_c - температура в точке установки клапана    

     calc_along_flow - направление расчета:  0 - против потока (расчет давления на входе);  1 - по потоку (расчет давления на выходе).    

     p_open_atma - давление открытия/закрытия клапана, атм    

     calibr - коэффициент калибровки   

        """

        self.f_GLV_p_atma = self.book.macro("GLV_p_atma")
        return self.f_GLV_p_atma(d_mm,p_calc_atma,q_gas_sm3day,gamma_g,t_C,calc_along_flow,p_open_atma,calibr)

    def unf_version(self, ):
        """
 ========== description ============== 
 функция возвращает номер версии Унифлок VBA 
        
 ==========  arguments  ============== 

     

        """

        self.f_unf_version = self.book.macro("unf_version")
        return self.f_unf_version()

    def decode_json(self, json,transpose=False,keys_filter="",only_values=False,safe_out=False):
        """
 ========== description ============== 
 Функция декодирования json строки,  позволяет вывести содержимое json строки в таблицу 
        
 ==========  arguments  ============== 

     json - строка содержащая результаты расчета    

     transpose - выбор вывода в строки или в столбцы    

     keys_filter - строка с ключами, которые надо вывести    

     only_values - если = 1 подписи выводиться не будут    

     safe_out - флаг заставляет выводить массив сторок,  что может работать лучше в офисе 2016 и ранее   

        """

        self.f_decode_json = self.book.macro("decode_json")
        return self.f_decode_json(json,transpose,keys_filter,only_values,safe_out)

    def decode_json_crv(self, json,transpose=False):
        """
 ========== description ============== 
 Функция декодирования json строки с табличной кривой,  позволяет вывести содержимое json строки в таблицу (на лист) 
        
 ==========  arguments  ============== 

     json - строка содержащая результаты расчета    

     transpose - выбор вывода в строки или в столбцы   

        """

        self.f_decode_json_crv = self.book.macro("decode_json_crv")
        return self.f_decode_json_crv(json,transpose)

    def encode_PVT(self, gamma_gas=const_gg_,gamma_oil=const_go_,gamma_wat=const_gw_,rsb_m3m3=const_rsb_default,pb_atma=-1,t_res_C=-1,bob_m3m3=-1,muob_cP=-1,PVT_corr_set=0):
        """
 ========== description ============== 
 Функция кодирования параметров PVT в строку,  для передачи PVT свойств в прикладные функции Унифлок. 
        
 ==========  arguments  ============== 

     gamma_gas - удельная плотность газа, по воздуху.  по умолчанию const_gg_ = 0.6    

     gamma_oil - удельная плотность нефти, по воде.  по умолчанию const_go_ = 0.86    

     gamma_wat - удельная плотность воды, по воде.  по умолчанию const_gw_ = 1    

     rsb_m3m3 - газосодержание при давлении насыщения, м3/м3.  по умолчанию const_rsb_default = 100  rp_m3m3 - замерной газовый фактор, м3/м3.  имеет приоритет перед rsb если rp < ..см.мануал   

     pb_atma - давление насыщения при температуре t_res_c, атма.  опциональный калибровочный параметр,  если не задан или = 0, то рассчитается по корреляции.    

     pb_atma - давление насыщения при температуре t_res_c, атма.  опциональный калибровочный параметр,  если не задан или = 0, то рассчитается по корреляции.  t_res_c - пластовая т..см.мануал   

     bob_m3m3 - объемный коэффициент нефти при давлении насыщения  и пластовой температуре, м3/м3.  по умолчанию рассчитывается по корреляции.    

     muob_cp - вязкость нефти при давлении насыщения.  и пластовой температуре, сп.  по умолчанию рассчитывается по корреляции.    

     pvt_corr_set - номер набора pvt корреляций для расчета:  0 - на основе корреляции стендинга;  1 - на основе кор-ии маккейна;  2 - на основе упрощенных зависимостей.   

        """

        self.f_encode_PVT = self.book.macro("encode_PVT")
        return self.f_encode_PVT(gamma_gas,gamma_oil,gamma_wat,rsb_m3m3,pb_atma,t_res_C,bob_m3m3,muob_cP,PVT_corr_set)

    def encode_feed(self, q_liq_sm3day=10,fw_perc=-1,rp_m3m3=-1,q_gas_free_sm3day=-1,fluid=PVT_DEFAULT):
        """
 ========== description ============== 
Функция кодирования параметров потока флюидов в строку, 
        
 ==========  arguments  ============== 

     q_liq_sm3day - дебит жидкости в ст.условиях.    

     fw_perc - ободненность, %    

     rp_m3m3 - газовый фактор, м3/м3:    

     q_gas_free_sm3day - расход свободного газа, ст. м3/сут    

   fluid  

        """

        self.f_encode_feed = self.book.macro("encode_feed")
        return self.f_encode_feed(q_liq_sm3day,fw_perc,rp_m3m3,q_gas_free_sm3day,fluid)

    def encode_ESP_pump_string(self, ESP_ID="1005",head_nom_m=2000,num_stages=0,freq_Hz=50,gas_correct=10,calibr=1,dnum_stages_integrate=1):
        """
 ========== description ============== 
 функция кодирования параметров работы УЭЦН в строку 
        
 ==========  arguments  ============== 

     esp_id - идентификатор насоса    

     head_nom_m - номинальный напор системы уэцн  - соответствует напора в записи эцн 50-2000    

     num_stages - количество ступеней, если задано  перекрывает значение напора    

     freq_hz - частота, гц  t_intake_c - температура на приеме насоа  t_dis_c - температура на выкиде насоса.  если = 0 и calc_along_flow = 1 то рассчитывается    

     gas_correct - деградация по газу:  0 - 2 задает значение вручную;  10 стандартный эцн (предел 25%);  20 эцн с газостабилизирующим модулем (предел 50%);  30 эцн с осевым модул..см.мануал   

     calibr - коэффициент поправки на напор.  если массив то второе значение - поправыка на подачу (множитель)  третье на мощность (множитель)    

     dnum_stages_integrate - шаг интегрирования для расчета   

        """

        self.f_encode_ESP_pump_string = self.book.macro("encode_ESP_pump_string")
        return self.f_encode_ESP_pump_string(ESP_ID,head_nom_m,num_stages,freq_Hz,gas_correct,calibr,dnum_stages_integrate)

    def encode_ESP_motor_string(self, motor_ID=0,U_surf_high_lin_V=2000,f_surf_Hz=50,power_fact_kW=100,U_nom_lin_V=1000,P_nom_kW=100,f_nom_Hz=50,eff_nom_fr=0.95,cosphi_nom_fr=0.9,slip_nom_fr=0.05,d_od_mm=90,Lambda=2,alpha0=1,xi0=1,Ixcf=1):
        """
 ========== description ============== 
 функция кодирования параметров ПЭД в строку 
        
 ==========  arguments  ============== 

    motor_id - тип 0 - постоянные значения,  1 - задается по каталожным кривым, ассинхронный  2 - задается по схеме замещения, ассинхронный    

    u_surf_high_lin_v - напряжение на поверхности  на высокой стороне трансформатора    

    f_surf_hz - частота питающего напряжения    

   power_fact_kw   

    u_nom_lin_v - номинальное напряжение двигателя, линейное, в    

    p_nom_kw - номинальная мощность двигателя квт    

    f_nom_hz - номинальная частота тока, гц    

    eff_nom_fr - кпд при номинальном режиме работы    

    cosphi_nom_fr - коэффициент мощности при номинальном режиме работы    

    slip_nom_fr - скольжение при номинальном режиме работы    

    d_od_mm - внешний диаметр - габарит пэд    

    lambda - для motorid = 2 перегрузочный коэффициент  отношение макс момента к номинальному    

    alpha0 - параметр. влияет на положение макс кпд.для motorid = 2    

    xi0 - параметр. определяет потери момента при холостом ходе.  для motorid = 2    

    ixcf - поправка на поправку тока холостого хода  при изменении напряжения и частоты от минимальной.   

        """

        self.f_encode_ESP_motor_string = self.book.macro("encode_ESP_motor_string")
        return self.f_encode_ESP_motor_string(motor_ID,U_surf_high_lin_V,f_surf_Hz,power_fact_kW,U_nom_lin_V,P_nom_kW,f_nom_Hz,eff_nom_fr,cosphi_nom_fr,slip_nom_fr,d_od_mm,Lambda,alpha0,xi0,Ixcf)

    def encode_ESP_cable_string(self, cable_R_Omkm,cable_X_Omkm,cable_t_max_C,manufacturer="no",name="no",d_mm=15,length_m=2000):
        """
 ========== description ============== 
 функция кодирования параметров  кабельной линии ПЭД в строку 
        
 ==========  arguments  ============== 

     cable_r_omkm - удельное активное сопротивление    

     cable_x_omkm - удельное реактивное сопротивление    

     cable_t_max_c - максимально допустимая температура    

     manufacturer - производитель, для справки    

     name - название кабеля, для справки    

     d_mm - диаметр жилы    

     length_m - длина кабельной линии, м   

        """

        self.f_encode_ESP_cable_string = self.book.macro("encode_ESP_cable_string")
        return self.f_encode_ESP_cable_string(cable_R_Omkm,cable_X_Omkm,cable_t_max_C,manufacturer,name,d_mm,length_m)

    def encode_ESP_separation_string(self, separation_mode,gassep_type,psep_man_atma,tsep_man_C,ksep_gassep_man_d,ksep_nat_man_d,ksep_liquid_man_d,M_Nm,length_m,manufacturer="no",name="no",natsep_type=0):
        """
 ========== description ============== 
 функция кодирования газосепаратора 
        
 ==========  arguments  ============== 

     separation_mode - режим расчета сепарации    

     gassep_type - тип - номер из базы    

     psep_man_atma - давление для расчета  коэффициента сепарации заданного вручную    

     tsep_man_c - температура для расчета  коэффициента сепарации заданного вручную    

     ksep_gassep_man_d - коэффициент сепарации гс заданный вручную    

     ksep_nat_man_d - коэффициент сепарации натуральной  заданный вручную    

     ksep_liquid_man_d - коэффициент сепарации жидкости для режима  потока через затруб    

     m_nm - момент на валу    

     length_m - длина кабельной линии, м    

     manufacturer - производитель, для справки    

     name - название кабеля, для справки  length_m - длина кабельной линии, м    

     natsep_type - модель расчета естественной сепарации  psep_man_atma - давление для расчета  коэффициента сепарации заданного вручную  tsep_man_c - температура для расчета  коэ..см.мануал  

        """

        self.f_encode_ESP_separation_string = self.book.macro("encode_ESP_separation_string")
        return self.f_encode_ESP_separation_string(separation_mode,gassep_type,psep_man_atma,tsep_man_C,ksep_gassep_man_d,ksep_nat_man_d,ksep_liquid_man_d,M_Nm,length_m,manufacturer,name,natsep_type)

    def encode_ambient_formation_string(self, therm_cond_form_WmC=2.4252,sp_heat_capacity_form_JkgC=200,therm_cond_cement_WmC=6.965,therm_cond_tubing_WmC=32,therm_cond_casing_WmC=32,heat_transfer_casing_liquid_Wm2C=200,heat_transfer_casing_gas_Wm2C=10,heat_transfer_fluid_convection_Wm2C=200,t_calc_hr=240):
        """
 ========== description ============== 
 функция кодирования температурных парамметров окружающей среды 
        
 ==========  arguments  ============== 

     therm_cond_form_wmc - теплопроводность породы окружающей среды    

     sp_heat_capacity_form_jkgc - удельная теплоемкость породы окружающей среды    

     therm_cond_cement_wmc - теплопроводность цементного камня вокруг скважины    

     therm_cond_tubing_wmc - теплопроводность стенок нкт    

   therm_cond_casing_wmc   

     heat_transfer_casing_liquid_wm2c - теплопередача через затруб с жидкостью    

     heat_transfer_casing_gas_wm2c - теплопередача через затруб с газом    

     heat_transfer_fluid_convection_wm2c - теплопередача в потоке  с жидкостью за счет конвекции    

     t_calc_hr - время на которое расчитывается распределение температуры   

        """

        self.f_encode_ambient_formation_string = self.book.macro("encode_ambient_formation_string")
        return self.f_encode_ambient_formation_string(therm_cond_form_WmC,sp_heat_capacity_form_JkgC,therm_cond_cement_WmC,therm_cond_tubing_WmC,therm_cond_casing_WmC,heat_transfer_casing_liquid_Wm2C,heat_transfer_casing_gas_Wm2C,heat_transfer_fluid_convection_Wm2C,t_calc_hr)

    def encode_well_construction_string(self, h_perf_m,h_tub_m,h_list_m,d_tub_list_mm,d_cas_list_mm,d_choke_mm,t_val_C,rough_m=0.0001):
        """
 ========== description ============== 
 функция кодирования параметров работы скважины с газлифтом 
        
 ==========  arguments  ============== 

    h_perf_m - глубина перфорации по длине скважины  точка узлового анализа для забоя    

    h_tub_m - глубина спуска нкт, или глубина  спуска эцн    

    h_list_m - траектория скважины, если число то измеренная  длина, range или таблица [0..n,0..1] то траектория    

    d_tub_list_mm - диаметр нкт. range или таблица [0..n,0..1]    

    d_cas_list_mm - диаметр эксп колонны.  range или таблица [0..n,0..1]    

    d_choke_mm - диаметр штуцера    

    t_val_c - температура вдоль скважины  если число то температура на устье скважины  если range или таблица [0..n,0..1] то температура  окружающей среды по вертикальной глубине, ..см.мануал   

    rough_m - шероховатость трубы   

        """

        self.f_encode_well_construction_string = self.book.macro("encode_well_construction_string")
        return self.f_encode_well_construction_string(h_perf_m,h_tub_m,h_list_m,d_tub_list_mm,d_cas_list_mm,d_choke_mm,t_val_C,rough_m)

    def encode_json(self, rng,always_collection=False):
        """
 ========== description ============== 
 функция кодирования диапазона ячеек в json строку 
        
 ==========  arguments  ============== 

     rng - диапазон ячеек для кодирования    

   always_collection  

        """

        self.f_encode_json = self.book.macro("encode_json")
        return self.f_encode_json(rng,always_collection)

    def encode_table_json(self, keyrange,val_namerange,valrange):
        """
 ========== description ============== 
 кодирование табличных результатов в json формат 
        
 ==========  arguments  ============== 

   keyrange   

   val_namerange   

   valrange  

        """

        self.f_encode_table_json = self.book.macro("encode_table_json")
        return self.f_encode_table_json(keyrange,val_namerange,valrange)

    def encode_pipe(self, h_list_m=1000,diam_list_mm=62,roughness_m=0):
        """
 ========== description ============== 
 encodes pipe construction into json string 
        
 ==========  arguments  ============== 

     h_list_m - trajectory data    

     diam_list_mm - массив для диаметров    

     roughness_m - значение шероховатости   

        """

        self.f_encode_pipe = self.book.macro("encode_pipe")
        return self.f_encode_pipe(h_list_m,diam_list_mm,roughness_m)

    def encode_t_model(self, t_model=StartEndTemp,t_list_C=50,t_start_C=-100,t_end_C=-100,param=""):
        """
 ========== description ============== 
 кодирование параметров температурной модели трубы/скважины 
        
 ==========  arguments  ============== 

     t_model - номер температурной модели    

     t_list_c - массив n*2 распределения температуры    

     t_start_c - температура в начале трубы    

     t_end_c - температура в конце трубы    

     param - параметры температурной модели  список параметров в мануале   

        """

        self.f_encode_t_model = self.book.macro("encode_t_model")
        return self.f_encode_t_model(t_model,t_list_C,t_start_C,t_end_C,param)

    def well_ksep_natural_d(self, q_liq_sm3day,fw_perc,p_intake_atma,t_intake_C=50,d_intake_mm=90,d_cas_mm=120,str_PVT=PVT_DEFAULT):
        """
 ========== description ============== 
 расчет натуральной сепарации газа на приеме насоса 
        
 ==========  arguments  ============== 

     q_liq_sm3day - дебит жидкости в поверхностных условиях    

     fw_perc - обводненность    

     p_intake_atma - давление сепарации    

     t_intake_c - температура сепарации    

     d_intake_mm - диаметр приемной сетки    

     d_cas_mm - диаметр эксплуатационной колонны    

     str_pvt - закодированная строка с параметрами pvt.  если задана - перекрывает другие значения   

        """

        self.f_well_ksep_natural_d = self.book.macro("well_ksep_natural_d")
        return self.f_well_ksep_natural_d(q_liq_sm3day,fw_perc,p_intake_atma,t_intake_C,d_intake_mm,d_cas_mm,str_PVT)

    def well_ksep_total_d(self, SepNat,SepGasSep):
        """
 ========== description ============== 
 расчет общей сепарации на приеме насоса 
        
 ==========  arguments  ============== 

     sepnat - естественная сепарация    

     sepgassep - искусственная сепарация (газосепаратор)  well_ksep_total_d = sepnat + (1 - sepnat) * sepgassep end function   

        """

        self.f_well_ksep_total_d = self.book.macro("well_ksep_total_d")
        return self.f_well_ksep_total_d(SepNat,SepGasSep)

    def crv_interpolation(self, x_points,y_points,x_val,type_interpolation=0):
        """
 ========== description ============== 
 функция поиска значения функции по заданным табличным данным (интерполяция) 
        
 ==========  arguments  ============== 

     x_points - таблица аргументов функции    

     y_points - таблица значений функции  количество агрументов и значений функции должно совпадать  для табличной функции одному аргументу соответствует  строго одно значение функ..см.мануал   

     x_val - аргумент для которого надо найти значение  одно значение в ячейке или диапазон значений  для диапазона аргументов будет найден диапазон значений  диапазоны могут быть ..см.мануал   

     type_interpolation - тип интерполяции  0 - линейная интерполяция  1 - кубическая интерполяция  2 - интерполяция акима (выбросы)  www.en.wikipedia.org/wiki/akima_spline  3 - ..см.мануал  

        """

        self.f_crv_interpolation = self.book.macro("crv_interpolation")
        return self.f_crv_interpolation(x_points,y_points,x_val,type_interpolation)

    def crv_interpolation_2D(self, XA,YA,FA,XYIA,out=1,type_interpolation=0):
        """
 ========== description ============== 
 функция поиска значения функции по двумерным табличным данным (интерполяция 2D) 
        
 ==========  arguments  ============== 

     xa - x значения исходных данных (строка значений или массив)    

     ya - y значения исходных данных (столбец значений или массив)    

     fa - табличные значения интерполируемой функции,  двумерная таблица или массив    

     xyia - таблица значений для которой надо найти результат  два столбца значений (x,y) или массив с двумя колонками  если не заданы возвращаются кубические коэффициента  для каж..см.мануал   

     out - для интерполяции кубическими сплайнами  out = 0 возвращаются только значения  out = 1 возвращаются значения и производные    

     type_interpolation - тип интерполяции  0 - линейная интерполяция  1 - кубическая интерполяция   

        """

        self.f_crv_interpolation_2D = self.book.macro("crv_interpolation_2D")
        return self.f_crv_interpolation_2D(XA,YA,FA,XYIA,out,type_interpolation)

    def crv_solve(self, x_points,y_points,y_val):
        """
 ========== description ============== 
 функция решения уравнения в табличном виде f(x) = y_val  ищется значение аргумента соответствующее заданному значению  используется линейная интерполяция  возможно несколько решений 
        
 ==========  arguments  ============== 

     x_points - таблица аргументов функции    

     y_points - таблица значений функции  количество агрументов и значений функции должно совпадать  для табличной функции одному аргументу соответствует  строго одно значение функ..см.мануал   

     y_val - значение функции для которого надо ищутся аргументы  строго одно вещественное число (ссылка на ячейку)   

        """

        self.f_crv_solve = self.book.macro("crv_solve")
        return self.f_crv_solve(x_points,y_points,y_val)

    def crv_intersection(self, x1_points,y1_points,x2_points,y2_points):
        """
 ========== description ============== 
Поиск пересечений для кривых заданных таблицами. Используется линейная интерполяция. Возможно несколько решений. 
        
 ==========  arguments  ============== 

     x1_points - таблица аргументов функции 1    

     y1_points - таблица значений функции 1  количество агрументов и значений функции должно совпадать  для табличной функции одному аргументу соответствует  строго одно значение ф..см.мануал   

     x2_points - таблица аргументов функции 2    

     y2_points - таблица значений функции 2  количество агрументов и значений функции должно совпадать  для табличной функции одному аргументу соответствует  строго одно значение ф..см.мануал  

        """

        self.f_crv_intersection = self.book.macro("crv_intersection")
        return self.f_crv_intersection(x1_points,y1_points,x2_points,y2_points)

    def crv_fit_spline_1D(self, XA,YA,M,XIA,WA,XCA,YCA,DCA,hermite=False):
        """
 ========== description ============== 
Поиск пересечений для кривых заданных таблицами. Используется линейная интерполяция. Возможно несколько решений. 
        
 ==========  arguments  ============== 

     xa - x значения исходных данных (строка значений или массив)    

     ya - y значения исходных данных (столбец значений или массив)  м - количество точек для сплайна интерполяции    

     должно быть четное для hermite = true    

     xia - таблица выходных значений  столбц значений (x) или массив в возрастающем порядке  если не заданы возвращаются кубические коэффициента для сегментов    

     wa - веса исходных данных    

     xca - х значения матрицы ограничений (столбец или массив)    

     yca - величина ограничения для заданного значения  (столбец или массив)    

     dca - тип ограничения. 0 - значение, 1 - наклон.  (столбец или массив).  если хоть одно из ограничений не задано - они не учитываются    

     должно быть четное для hermite = true  xia - таблица выходных значений  столбц значений (x) или массив в возрастающем порядке  если не заданы возвращаются кубические коэффицие..см.мануал  

        """

        self.f_crv_fit_spline_1D = self.book.macro("crv_fit_spline_1D")
        return self.f_crv_fit_spline_1D(XA,YA,M,XIA,WA,XCA,YCA,DCA,hermite)

    def crv_fit_linear(self, YA,XA,out,weight,constraints):
        """
 ========== description ============== 
Аппроксимация данных линейной функцией. Решается задача min|XM-Y| ищется вектор M 
        
 ==========  arguments  ============== 

     ya - y вектор исходных данных [0..n-1] (столбец или массив)    

     xa - x матрица исходных данных [0..n-1, 0..d-1]  (таблица или массив)    

     out - тип вывода,  out=0 (по умолчанию) коэффициенты аппроксимации [0..d-1],  out=1 код ошибки подбора аппроксимации  out=2 отчет по подбору аппроксимации,  avgerror, avgrele..см.мануал   

     weight - вектор весов [0..n-1] для каждого параметра    

     constraints - матрица ограничений с [0..k-1, 0..d] такая что  c[i,0]*m[0] + ... + c[i,d-1]*c[d-1] = cmatrix[i,d]   

        """

        self.f_crv_fit_linear = self.book.macro("crv_fit_linear")
        return self.f_crv_fit_linear(YA,XA,out,weight,constraints)

    def crv_fit_poly(self, YA,XA,M,out,XIA,weight,constraints):
        """
 ========== description ============== 
Аппроксимация данных полиномом функцией. Решается задача min|XM-Y| ищется вектор M 
        
 ==========  arguments  ============== 

     ya - y вектор исходных данных [0..n-1] (столбец или массив)    

     xa - х вектор исходных данных [0..n-1] (таблица или массив)    

     m - степень полинома для аппроксимации    

     out - тип вывода, out=0 (по умолчанию) значения полинома для xia,  out=1 код ошибки аппроксимации  out=2 отчет по подбору аппроксимации,  avgerror, avgrelerror, maxerror, rmse..см.мануал   

     out - тип вывода, out=0 (по умолчанию) значения полинома для xia,  out=1 код ошибки аппроксимации  out=2 отчет по подбору аппроксимации,  avgerror, avgrelerror, maxerror, rmse..см.мануал   

     weight - вектор весов [0..n-1] для каждого параметра    

     constraints - матрица ограничений с[0..k-1,0..2].  с[i,0] - значение x где задано ограничение  с[i,1] - велична ограничения,  с[i,2] - тип ограничения (0 -значение,1 -производ..см.мануал  

        """

        self.f_crv_fit_poly = self.book.macro("crv_fit_poly")
        return self.f_crv_fit_poly(YA,XA,M,out,XIA,weight,constraints)

    def crv_parametric_interpolation(self, x_points,y_points,x_val,type_interpolation=0,param_points=-1):
        """
 ========== description ============== 
 интерполяция функции заданной параметрически  параметр номер значения 
        
 ==========  arguments  ============== 

     x_points - таблица аргументов функции    

     y_points - таблица значений функции  количество агрументов и значений функции должно совпадать  для табличной функции одному аргументу соответствует  строго одно значение функ..см.мануал   

     x_val - аргумент для которого надо найти значение  одно значение в ячейке или диапазон значений  для диапазона аргументов будет найден диапазон значений  диапазоны могут быть ..см.мануал   

     type_interpolation - тип интерполяции  0 - линейная интерполяция  1 - кубическая интерполяция  2 - интерполяция акима (выбросы)  www.en.wikipedia.org/wiki/akima_spline  3 - ..см.мануал   

   param_points  

        """

        self.f_crv_parametric_interpolation = self.book.macro("crv_parametric_interpolation")
        return self.f_crv_parametric_interpolation(x_points,y_points,x_val,type_interpolation,param_points)

    def Ei(self, x):
        """
 ========== description ============== 
 Расчет интегральной показательной функции Ei(x) 
        
 ==========  arguments  ============== 

     x - агрумент функции, может быть и положительным и отрицательным   

        """

        self.f_Ei = self.book.macro("Ei")
        return self.f_Ei(x)

    def E_1(self, x):
        """
 ========== description ============== 
 Расчет интегральной показательной функции $E_1(x)$  для вещественных положительных x, x>0 верно E_1(x)=- Ei(-x) 
        
 ==========  arguments  ============== 

     x - агрумент функции, может быть и положительным и отрицательным   

        """

        self.f_E_1 = self.book.macro("E_1")
        return self.f_E_1(x)

    def transient_pd_radial(self, td,cd=0,skin=0,rd=1,Model=0):
        """
 ========== description ============== 
 Расчет неустановившегося решения уравнения фильтрации  для различных моделей радиального притока к вертикльной скважине  основано не решениях в пространстве Лапласа и преобразовании Стефеста 
        
 ==========  arguments  ============== 

     td - безразмерное время для которого проводится расчет  сd - безразмерный коэффициент влияния ствола скважины    

   cd   

     skin - скин-фактор, безразмерный skin>0.  для skin<0 используйте эффективный радиус скважины    

     rd - безразмерное расстояние для которого проводится расчет  rd=1 соответвует забою скважины    

     model - модель проведения расчета. 0 - модель линейного стока ei  1 - модель линейного стока через преобразование стефеста  2 - конечный радиус скважины  3 - линейный сток со ..см.мануал  

        """

        self.f_transient_pd_radial = self.book.macro("transient_pd_radial")
        return self.f_transient_pd_radial(td,cd,skin,rd,Model)

    def transient_pwf_radial_atma(self, t_hr,q_liq_sm3day,pi_atma=250,skin=0,cs_1atm=0,r_m=0.1,rw_m=0.1,k_mD=100,h_m=10,porosity=0.2,mu_cP=1,b_m3m3=1.2,ct_1atm=0.00001,Model=0):
        """
 ========== description ============== 
 расчет изменения забойного давления после запуска скважины  с постоянным дебитом (terminal rate solution) 
        
 ==========  arguments  ============== 

     t_hr - время для которого проводится расчет, час    

     q_liq_sm3day - дебит запуска скважины, м3/сут в стандартных условиях    

     pi_atma - начальное пластовое давление, атма    

     skin - скин - фактор, может быть отрицательным    

     cs_1atm - коэффициент влияния ствола скважины, 1/атм    

     r_m - расстояние от скважины для которого проводится расчет, м    

     rw_m - радиус скважины, м    

     k_md - проницаемость пласта, мд    

     h_m - толщина пласта, м    

     porosity - пористость    

     mu_cp - вязкость флюида в пласте, сп    

     b_m3m3 - объемный коэффициент нефти, м3/м3    

     ct_1atm - общая сжимаемость системы в пласте, 1/атм    

     model - модель проведения расчета. 0 - модель линейного стока ei  1 - модель линейного стока через преобразование стефеста  2 - конечный радиус скважины  3 - линейный сток со ..см.мануал  

        """

        self.f_transient_pwf_radial_atma = self.book.macro("transient_pwf_radial_atma")
        return self.f_transient_pwf_radial_atma(t_hr,q_liq_sm3day,pi_atma,skin,cs_1atm,r_m,rw_m,k_mD,h_m,porosity,mu_cP,b_m3m3,ct_1atm,Model)

    def transient_def_cd(self, cs_1atm,rw_m=0.1,h_m=10,porosity=0.2,ct_1atm=0.00001):
        """
 ========== description ============== 
 расчет безразмерного коэффициента влияния ствола скважины (определение) 
        
 ==========  arguments  ============== 

     cs_1atm - коэффициент влияния ствола скважины, 1/атм    

     rw_m - радиус скважины, м    

     h_m - толщина пласта, м    

     porosity - пористость    

     ct_1atm - общая сжимаемость системы в пласте, 1/атм   

        """

        self.f_transient_def_cd = self.book.macro("transient_def_cd")
        return self.f_transient_def_cd(cs_1atm,rw_m,h_m,porosity,ct_1atm)

    def transient_def_cs_1atm(self, cd,rw_m=0.1,h_m=10,porosity=0.2,ct_1atm=0.00001):
        """
 ========== description ============== 
 расчет коэффициента влияния ствола скважины (определение) 
        
 ==========  arguments  ============== 

   cd   

     rw_m - радиус скважины, м    

     h_m - толщина пласта, м    

     porosity - пористость    

     ct_1atm - общая сжимаемость системы в пласте, 1/атм   

        """

        self.f_transient_def_cs_1atm = self.book.macro("transient_def_cs_1atm")
        return self.f_transient_def_cs_1atm(cd,rw_m,h_m,porosity,ct_1atm)

    def transient_def_td(self, t_day,rw_m=0.1,k_mD=100,porosity=0.2,mu_cP=1,ct_1atm=0.00001):
        """
 ========== description ============== 
 расчет безразмерного времени (определение) 
        
 ==========  arguments  ============== 

     t_day - время для которого проводится расчет, сут    

     rw_m - радиус скважины, м    

     k_md - проницаемость пласта, мд    

     porosity - пористость    

     mu_cp - вязкость флюида в пласте, сп    

     ct_1atm - общая сжимаемость системы в пласте, 1/атм   

        """

        self.f_transient_def_td = self.book.macro("transient_def_td")
        return self.f_transient_def_td(t_day,rw_m,k_mD,porosity,mu_cP,ct_1atm)

    def transient_def_t_day(self, td,rw_m=0.1,k_mD=100,porosity=0.2,mu_cP=1,ct_1atm=0.00001):
        """
 ========== description ============== 
 расчет времени по безразмерному времени (определение) 
        
 ==========  arguments  ============== 

   td   

     rw_m - радиус скважины, м    

     k_md - проницаемость пласта, мд    

     porosity - пористость    

     mu_cp - вязкость флюида в пласте, сп    

     ct_1atm - общая сжимаемость системы в пласте, 1/атм   

        """

        self.f_transient_def_t_day = self.book.macro("transient_def_t_day")
        return self.f_transient_def_t_day(td,rw_m,k_mD,porosity,mu_cP,ct_1atm)

    def transient_def_pd(self, p_wf_atma,q_liq_sm3day,pi_atma=250,k_mD=100,h_m=10,mu_cP=1,b_m3m3=1.2):
        """
 ========== description ============== 
 расчет безразмерного давления (определение) 
        
 ==========  arguments  ============== 

     p_wf_atma - забойное давление, атма    

     q_liq_sm3day - дебит запуска скважины, м3/сут в стандартных условиях    

     pi_atma - начальное пластовое давление, атма    

     k_md - проницаемость пласта, мд    

     h_m - толщина пласта, м    

     mu_cp - вязкость флюида в пласте, сп    

     b_m3m3 - объемный коэффициент нефти, м3/м3   

        """

        self.f_transient_def_pd = self.book.macro("transient_def_pd")
        return self.f_transient_def_pd(p_wf_atma,q_liq_sm3day,pi_atma,k_mD,h_m,mu_cP,b_m3m3)

    def transient_def_p_wf_atma(self, pd,q_liq_sm3day,pi_atma=250,k_mD=100,h_m=10,mu_cP=1,b_m3m3=1.2):
        """
 ========== description ============== 
 расчет безразмерного давления (определение) 
        
 ==========  arguments  ============== 

   pd   

     q_liq_sm3day - дебит запуска скважины, м3/сут в стандартных условиях    

     pi_atma - начальное пластовое давление, атма    

     k_md - проницаемость пласта, мд    

     h_m - толщина пласта, м    

     mu_cp - вязкость флюида в пласте, сп    

     b_m3m3 - объемный коэффициент нефти, м3/м3   

        """

        self.f_transient_def_p_wf_atma = self.book.macro("transient_def_p_wf_atma")
        return self.f_transient_def_p_wf_atma(pd,q_liq_sm3day,pi_atma,k_mD,h_m,mu_cP,b_m3m3)

#UniflocVBA = API(addin_name_str)
