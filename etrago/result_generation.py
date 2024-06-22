import etrago

n = etrago.Etrago(
    csv_folder_name = "D:\eTraGo\etrago\Result-V06 500h-100C",
    json_path = "D:\eTraGo\etrago\Result-V06 500h-100C.json"
)

# n.plot_carrier()
# n.plot_carrier(carrier_links='power_to_Hydrogen', carrier_buses='AC')

# n.plot_clusters(carrier = 'O2',save_path="D:\plot.png")

# n.plot_flexibility_usage(flexibility=, agg="5h", snapshots=[], buses=[], pre_path=None, apply_on="grid_model")
# n.plot_flexibility_usage(flexibility='heat',save_path="D:\flexibility_usage_heat_w.png")
# flexibility='dsm'/'BEV charger'/battery/heat/h2_store)

# n.plot_gas_generation() #this generate a liner graph to show production for specific period by GW

# n.plot_gas_summary()
# n.plot_grid(line_colors= 'expansion_abs/line_loading'/expansion_rel/v_nom)
# n.plot_h2_generation()
# n.plot_h2_summary()results300hours/
# n.plot_heat_loads()
# n.plot_heat_summary()
# b.plot_gas_summary(t_resolution='300H', stacked=False, save_path=False )

# ssh key public and private and having ip address of the server 

# n.network.lopf(pyomo=True, solver_name="gurobi", extra_functionality=add_wwtp_constraints)
# n.network.model.power_constraint_power_ratio0 #line 3123 in constraints


n.plot_h2_generation()
# #below command lines are to find/work with time series data
# n.network.generators_t
# n.network.loads_t.p
# n.loads_t.p
# n.network.loads_t.p.iloc[1]
# n.network.loads_t.p.iloc[1]
# n.network.loads_t.p.iloc[:1]
# n.network.loads_t.p.iloc[:1].plot()
# n.network.loads_t.p.iloc[:1].T.plot()
# # compere the new version of plots with dev branch

# n.network.links[n.network.links.carrier=='power_to_O2'].p_nom_opt
# n.network.links[n.network.links.carrier=='power_to_O2'].p_nom
# n.network.links_t.p0['index number'].max()
# n.network.links_t.p0['index number'].min()


