from enum import StrEnum

class Column(StrEnum):
    SCN_NAME = "scn_name"
    # ID = "scn_name"
    ID_WWTP =   "WWTP ID"
    ID_H2 =     "H2 ID"
    ID_AC =     "AC ID"
    ID_HEAT =   "Heat ID"
    ID_KA =     "KA ID"
    ID_OPTIMAL ="Optimal ID"

    ID_P2H2 =   "link ID Power to H2"
    ID_H22P =   "link ID H2 to Power"
    ID_H2P =    "link ID Heat to Power"
    ID_O22P =   "link ID O2 to Powre"

    LINK_AC =   "Link to AC"
    LINK_H2 =   "Link to H2"
    LINK_HEAT = "Link to Heat"
    LINK_O2 =   "Link to WWTP"

    TYPE_H2 =   "H2 Type"
    TYPE_AC =   "AC Type"

    POINT_WWTP =    "Point WWTP"
    POINT_H2 =      "Point H2"
    POINT_HEAT =    "Point Heat"
    POINT_AC =      "Point AC"
    POINT_OPTIMAL = "Point Optimal"

    DISTANCE_AC =   "Distance to AC [km]"
    DISTANCE_HEAT = "Distance to Heat [km]"
    DISTANCE_H2 =   "Distance to H2 [km]"
    DISTANCE_O2 =   "Distance to WWTP [km]"

    WWTP_NAME =     "WWTP Name"
    WWTP_CLASS =    "WWTP Class"
    WWTP_PE =       "WWTP Capacity [PE]"
    WWTP_EC =       "WWTP EC [MWh/year]"
    WWTP_EC_H =     "WWTP EC [MWh/hour]"
    WWTP_AER_EC =   "WWTP Aeration EC [MWh/year]"
    WWTP_O2_EC_H =  "O2 to Power [MWh/hour]"
    DEMAND_O2 =     "O2 Demand 2035 [tonne/year]"
    DEMAND_O3 =     "O3 Demand 2035 [tonne/year]"
    DEMAND_O2_TOTAL="Total O2 Demand [kgO2/hour]"		#Pure O2 demand (O2+O3)
    O2_SELL_PRICE = "O2 Selling price [Euro/kgO2]"
    O2_H2_SELL =    "O2 sellable [Euro/kgH2]"
    PIPELINE_DIAMETER_O2 = "O2 pipeline diameter [m]"

    FLOW_RATE_O2 =          "flow rate O2 [m3/hour]"
    FLOW_RATE_H2 =          "flow rate H2 [m3/hour]"
    PRESSURE_H2_FINAL =     "H2 final pressure [bar]"
    PRESSURE_O2_FINAL =     "O2 final pressure [bar]"
    PIPELINE_DIAMETER_H2 =  "H2 pipeline diameter [m]"

    H2_PROD_Y =     "Hydrogen Production [kgH2/year]"
    H2_PROD_H =     "Hydrogen Production [kgH2/hour]"
    ELZ_CAP =       "Electrolyzer capacity [MW]",
    ELZ_EFF =       "Electrolyzer Efficiency"
    H2_EFF =        "H2-to-power efficiency"

    CAPEX_AC =      "AC CAPEX [Euro/MW/year]"
    CAPEX_H2 =      "H2 CAPEX [Euro/MW/year]"
    CAPEX_HEAT =    "Heat CAPEX [Euro/MW/year]"
    CAPEX_O2 =      "O2 CAPEX [Euro/MW/year]"
    LCOH =          "LCOH [Euro/kgH2]"
    LCOH_O2_HEAT =  "LCOH +O2 +Heat"
    LCOH_O2 =       "LCOH +O2"
    LCOH_HEAT =     "LCOH +Heat"

    POWER_TO_H2 =       "power to H2 [MWh/hour]"
    H2_TO_POWER =       "H2 to power [MWh/hour]"
    POWER_TO_O2 =       "power to O2 [MWh/hour]"
    POWER_TO_HEAT =     "power to Heat [MWh/hour]"

    ETRAGO_CAPEX_PTH2 =  "eTraGo AC CAPEX [Euro/MW/year]"
    ETRAGO_CAPEX_H2TP =  "eTraGo H2 CAPEX [Euro/MW/year]"
    ETRAGO_CAPEX_PTO2 =  "eTraGo O2 CAPEX [Euro/MW/year]"
    ETRAGO_CAPEX_PTHEAT ="eTraGo Heat CAPEX [Euro/MW/year]"

    ETRAGO_EFF_PTH2 = "power to H2 effeciency"
    ETRAGO_EFF_H2TP = "H2 to power effeciency"
    ETRAGO_EFF_PTO2 = "O2 to power effeciency"
    ETRAGO_EFF_PTHEAT = "heat to power effeciency"

    ETRAGO_PNOM_PTH2 = "p_nom power to H2 [MWH/HOUR]"
    ETRAGO_PNOM_H2TP = "p_nom H2 to power [MWH/HOUR]"
    ETRAGO_PNOM_PTO2 = "p_nom O2 to power [MWH/HOUR]"
    ETRAGO_PNOM_PTHEAT = "p_nom Heat to power [MWH/HOUR]"

    TOTAL_CAPEX_PTH2 =  "Total AC CAPEX [Euro/year]"
    TOTAL_CAPEX_H2TP =  "Total H2 CAPEX [Euro/year]"
    TOTAL_CAPEX_PTO2 =  "Total O2 CAPEX [Euro/year]"
    TOTAL_CAPEX_PTHEAT ="Total Heat CAPEX [Euro/year]"

    ETRAGO_TYPE_PTH2 ="etrago_type_power_to_h2"
    ETRAGO_TYPE_H2TP = "etrago_type_h2_to_power"
    ETRAGO_TYPE_PTO2 = "etrago_type_power_to_O2"
    ETRAGO_TYPE_PTHEAT = "etrago_type_power_to_heat"	


