# esp_design
Подбор ЭЦН с учетом неопределенности в исходных данных

расчетный модуль для магистерской диссертации

# Анализ работы скважин и скважинного оборудования с учетом неопределенности в исходных данных для условий месторождений Западной Сибири

Кобзарь О.С., Хабибуллин Р.А., 2021 

## структура репозитория
* UniflocVBA - фиксированные версии UniflocVBA 7.25 и 7.28, использованные в расчете
* well_model - модуль со всеми функциями по расчету физической модели, расчету узлового анализа,
  применения метода Монте-Карло
* runner - функция-интерфейс для вызова основного pipeline 
* mult - запуск pipeline в режиме многопоточности
* esp_design_m - основная рабочая тетрадка для подбора ЭЦН, использующая функции из well_model
* design_analysis - тетрадка для анализа
* calc_data - результаты расчетов, используемые в диссертации

