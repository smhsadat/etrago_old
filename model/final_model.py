import pandas as pd
import math
import geopandas as gpd
from itertools import count
from rtree import index
from geopandas.tools import sjoin
from shapely.ops import nearest_points
from sqlalchemy import create_engine, text
from scipy.optimize import minimize
from shapely.io import from_wkt, to_wkt
from shapely.geometry import MultiLineString, LineString, Point
from columns import Column
from shapely.wkb import dumps

# Input Constant Parameters 
    # General input data for the Model
SCENARIO_NO=2                   # 1 = WWTP-based location, 2 = AC-based location
OPTIMIZATION = "no"	            # yes or no to activate optimization of for the optimal location
SUBSTATION = "yes"              # yes or no will switch between AC points and Substation points.
SCENARIO_NAME = "eGon2035"      # scenario name same as eTraGo database scenario
ID_OPTIMAL_START = 80_000      
DATA_CRS=4326
METRIC_CRS=3857
DISCOUNT_RATE = 0.05

H2 = 'h2'
WWTP = 'wwtp'
AC = 'ac'
H2GRID = 'h2_grid'
ACZONE = 'ac_zone'
ACSUB = "ac_sub"
O2 = 'o2'
HEAT = 'heat_point'

MAXIMUM_DISTANCE = { # meter
    O2: 600,
    AC: 80000 + 100,
    H2: 80000 + 100,
    HEAT: 80000 + 100,
}

# Electricity
ELEC_COST = 60 							# [EUR/MWh]
AC_TRANS = 17_500			            # [EUR/MVA]
AC_LIFETIME_CABLE = 25					# [year]
AC_COST_CABLE = 800_000					# [EUR/km/MVA]
FUEL_CELL_EFF = 0.5						# to use as effeciency of hydrogen to power

# Heat
HEAT_RATIO = 0.2						# % of the total kwh of electrolyzer production maybe the energy kwh in H2
HEAT_LIFETIME = 20						# [YEAR]
HEAT_EFFICIENCY = 0.8805				# [] according to Lennart data
HEAT_COST_PIPELINE = 25_000				# [EUR/MWH/KM] overal cost for heat except pipeline
HEAT_SELLING_PRICE = 21.6				# [EUR/MWh]

# Wastewater Treatment Plants (WWTP)
WWTP_SEC = {'c5': 29.6, 'c4': 31.3,'c3': 39.8, 'c2': 42.1}	# [kWh/year] Specific Energy Consumption for different class of WWTPs
O2_O3_RATIO = 1.7 						# [-]
O2_H2_RATIO = 7.7						# [-]
O2_PURE_RATIO = 20.95/100				# [-]
FACTOR_AERATION_EC = 0.6				# [%] aeration Electrical Consumption from total capacity of WWTP (PE)
FACTOR_O2_EC = 0.8						# [%] Oxygen Electrical Consumption from total aeration EC
O2_LIFETIME_PIPELINE = 25				
O2_EFFICIENCY = 0.9
O2_PRESSURE_MIN = 2				     	# [bar]
O2_COST_EQUIPMENT = 5000
O2_LIEFTIME_EQUIPMENT = 25

# Electrolyzers (ELZ)
ELZ_SEC = 50	 						# [kWh/kgH2] electrolyzer specific energy 
ELZ_EFF = 33.33/ELZ_SEC					# [%] H2 energy kWh/kgH2 / electricity input kWh/kgH2
ELZ_FLH = 8760 							# [hour] full load hours 		5217
ELZ_LIFETIME_H = 85_000 / ELZ_FLH		# [Year] lifetime of stack [15 years]
ELZ_LIFETIME_Y = 25						# [Year] lifetiem of ELZ system in [year]
ELZ_CAPEX_SYSTEM = 504_000				# [EUR/MW]
ELZ_CAPEX_STACK = 180_000*2				# [EUR/MW] to extend the lifetime of the stack same as the lifetime of the system
ELZ_OPEX = (ELZ_CAPEX_SYSTEM+ELZ_CAPEX_STACK)*0.03	# [EUR/MW]	3% of total CAPEX per year
H2_TO_POWER_EFF = 0.5					# as per available postgres database
H2_PRESSURE_ELZ = 30					# [bar]		1.01325
O2_PRESSURE_ELZ = 13					# [bar]

# Hydrogen Pipeline
H2_PRESSURE_MIN = 29					# [bar]
H2_LIFETIME_PIPELINE = 25				# [YEAR]
H2_COST_PIPELINE = 25_000				# [EUR/MW]

# general gas pipeline constant
PIPELINE_DIAMETER_RANGE = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] # range of pipeline size [m]
TEMPERATURE = 15 + 273.15			    # [Kelvin] degree + 273.15
UNIVERSAL_GAS_CONSTANT = 8.3145			# [J/(mol·K)]
MOLAR_MASS_H2 = 0.002016				# [kg/mol]
MOLAR_MASS_O2 = 0.0319988				# [kg/mol]



# connet to PostgreSQL database (to server)
engine = create_engine(
    "postgresql+psycopg2://egon:data@localhost:59738/etrago-data",
    echo=False,
)

# # connet to PostgreSQL database (to localhost)
# engine = create_engine(
#     "postgresql+psycopg2://postgres:postgres@localhost:5432/etrago-data",
#     echo=False,
# )

# read and reproject spatial data
def read_query(engine, query):
    return gpd.read_postgis(query, engine, crs=DATA_CRS).to_crs(3857)


def export_to_db(df):
    max_bus_id = 77600
    next_bus_id = count(start=max_bus_id, step=1)
    table_name = "egon_etrago_bus"
    
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM grid.{table_name} WHERE bus_id >= {max_bus_id} AND carrier = 'O2'"))
    
    df = df.copy(deep=True)
    result = []
    for _, row in df.iterrows():
        bus_id = next(next_bus_id)
        result.append({
            "scn_name": 'eGon2035',
            "bus_id": bus_id,
            "v_nom": '110',
            "type": row['KA ID'],
            "carrier": 'O2',
            "x": row['longitude Kläranlage_rw'],
            "y": row['latitdue Kläranlage_hw'],
            "geom": dumps(Point(row['longitude Kläranlage_rw'], row['latitdue Kläranlage_hw']), srid=4326),
            "country": 'DE',
        })
    result_df = pd.DataFrame(result)
    result_df.to_sql(
        table_name,
        engine,
        schema="grid",
        if_exists="append",
        index=False
    )
wwtp_spec = pd.read_csv("WWTP_spec.csv", index_col=False)
export_to_db(wwtp_spec)  # Call the function with the dataframe
print("WWTP points exported to: egon_etrago_bus")


# dictionary of SQL queries
queries = {
    WWTP: """
            SELECT bus_id AS id, geom, type AS ka_id
            FROM grid.egon_etrago_bus
            WHERE carrier in ('O2')
            """,
    H2:  """
            SELECT bus_id AS id, geom 
            FROM grid.egon_etrago_bus
            WHERE carrier in ('CH4')
            AND scn_name = 'eGon2035'
            AND country = 'DE'
            """,
    H2GRID: """
            SELECT link_id AS id, geom 
            FROM grid.egon_etrago_link
            WHERE carrier in ('CH4') AND scn_name  = 'eGon2035'
            LIMIT 0
            """,
    AC:   """
            SELECT bus_id AS id, geom
            FROM grid.egon_etrago_bus
            WHERE carrier in ('AC')
            AND scn_name = 'eGon2035'
            AND v_nom = '110'
            """,
    
    ACSUB:   """
            SELECT bus_id AS id, point AS geom
            FROM grid.egon_hvmv_substation
            """,
    ACZONE: """
            SELECT bus_id AS id, ST_Transform(geom, 4326) as geom
            FROM grid.egon_mv_grid_district
            """,
	HEAT: """
			SELECT bus_id AS id, geom
			FROM grid.egon_etrago_bus
			WHERE carrier in ('central_heat')
            AND scn_name = 'eGon2035'
            AND country = 'DE'
            """,
}


# First Phase: Find intersection
    # Data management
    # read and convert the spatial CRS data to Metric CRS 
dfs = { key: gpd.read_postgis(queries[key], engine, crs=4326).to_crs(3857) for key in queries.keys() }

    # First Phase: Find intersection between points
    # Perform spatial join to find points within zones (substation zones)
in_zone = {
    'wwtp': sjoin(dfs[WWTP], dfs[ACZONE], how="inner", predicate='within'),
    'ac': sjoin(dfs[AC if SUBSTATION == "no" else ACSUB], dfs[ACZONE], how="inner", predicate='within'),
}

    # Create R-tree index to speedup the process based on bounding box coordinates.
rtree = { key: index.Index() for key in [H2, AC, ACSUB, H2GRID, HEAT] }
for key in rtree.keys():
    for i in range(len(dfs[key])):
        rtree[key].insert(i, dfs[key].iloc[i].geom.bounds)

    # Find the nearest intersection relation between AC points and WWTPs
     	# 1. find ACs inside same network zone as wwtp
     	# 2. calculate distances betweeen those acs and wwtp within a identical zone
     	# 3. select the point which has the minimum distance among them
     	# 4. distingush type of AC (point or substation)
def find_closest_acs():
	results = []    
    # Iterate over the zones and calculate distances
	for zone_id in dfs[ACZONE].index:
		wwtp_in_zones = in_zone[WWTP][in_zone[WWTP]['index_right'] == zone_id]
		ac_in_zones = in_zone[AC][in_zone[AC]['index_right'] == zone_id]
		for _, wwtp_row in wwtp_in_zones.iterrows():
			for _, ac_row in ac_in_zones.iterrows():
				distance = round(wwtp_row.geom.distance(ac_row.geom))
				results.append({
 					Column.ID_WWTP: wwtp_row['id_left'],
 					Column.ID_KA: wwtp_row['ka_id'],
 					Column.ID_AC: ac_row['id_left'],
 					Column.DISTANCE_AC: distance / 1000,	# km
 					Column.POINT_WWTP: wwtp_row.geom,
 					Column.POINT_AC: ac_row.geom,
				})
	results = pd.DataFrame(results).drop_duplicates()
	results = results.loc[results.groupby([Column.ID_WWTP])[Column.DISTANCE_AC].idxmin()]
	results = results[results[Column.DISTANCE_AC] < MAXIMUM_DISTANCE[AC]]
	return results
    # Creating the initial main dataframe 
main_df = find_closest_acs()

    # merge and find the AC and Substation type for AC points 
def find_ac_type(dataframe_with_ac):
    result = dataframe_with_ac.copy()
    def _find_ac_point(row):
        substations = dfs[ACSUB].loc[dfs[ACSUB]['id'] == row]
        points = dfs[AC].loc[dfs[AC]['id'] == row]
        is_sub = len(substations) > 0
        if is_sub:
            return substations.iloc[0]["geom"]
        else:
            return points.iloc[0]["geom"]
    def _find_ac_type(row):
        substations = dfs[ACSUB].loc[dfs[ACSUB]['id'] == row]
        is_sub = len(substations) > 0
        if is_sub:
            return "substation"
        else:
            return "ac_point"
    result[Column.POINT_AC] = main_df[Column.ID_AC].apply(_find_ac_point)
    result[Column.TYPE_AC] = main_df[Column.ID_AC].apply(_find_ac_type)
    return result

main_df = find_ac_type(main_df)
main_df.to_csv("O2_AC.csv", index=False)

    # The function find and assign the correct reference point for centrlizing as buffer for further steps
def get_main_point():
	if SCENARIO_NO == 1:
		return Column.ID_WWTP, Column.POINT_WWTP
	elif SCENARIO_NO == 2:
		return Column.ID_AC, Column.POINT_AC
	else:
		raise Exception("Invalid scenario number")

    # Find nearest H2 points & grid pipeline to refernce points (AC or WWTP depend on scenario no)
    # below function support h2 points and h2_grid, by distingushing their types
def find_h2_intersections(rtree, df, buffer_factor, type):
	results = []
	col, point = get_main_point()
	for _, row in main_df.iterrows():
		buffered = row[point].buffer(buffer_factor)
		for idx in rtree.intersection(buffered.bounds):
			item = df.iloc[idx]
			if buffered.intersects(item.geom):
				distance = round(row[point].distance(item.geom))
				near_point = nearest_points(item.geom, row[point])[0]
				results.append({
 					col: row[col],
 					Column.ID_H2: item.id,
 					Column.DISTANCE_H2: distance/1000,
 					Column.POINT_H2: near_point,
 					Column.TYPE_H2: type,
				})
	return pd.DataFrame(results)
h2_intersections = find_h2_intersections(rtree[H2], dfs[H2], MAXIMUM_DISTANCE[H2], H2)
h2_grid_intersections = find_h2_intersections(rtree[H2GRID], dfs[H2GRID], MAXIMUM_DISTANCE[H2], H2GRID)

def find_minimum_h2_intersections():
	col, _ = get_main_point()
	union = pd.concat([h2_intersections, h2_grid_intersections]).reset_index(drop=True)
	result = union.iloc[union.groupby(col)[Column.DISTANCE_H2].idxmin()]
	return result
min_h2_intersections = find_minimum_h2_intersections()
min_h2_intersections.to_csv("Ref_H2.csv", index=False)

    # Find nearest Heat Points to refernce points
def find_heatpoint_intersections(rtree):
	col, point = get_main_point()
	results = []
	for _, row in main_df.iterrows():
		buffered = row[point].buffer(MAXIMUM_DISTANCE[HEAT])
		for idx in rtree.intersection(buffered.bounds):
			item = dfs[HEAT].iloc[idx]
			if buffered.intersects(item.geom):
				distance = round(row[point].distance(item.geom))
				results.append({
 					col: row[col],
 					Column.ID_HEAT: item.id,
 					Column.DISTANCE_HEAT: distance/1000,
					Column.POINT_HEAT: item.geom,
				})
	return pd.DataFrame(results)

heatpoint_intersections = find_heatpoint_intersections(rtree[HEAT])

def find_minimum_heatpoint_intersections():
	col, _ = get_main_point()
	result = heatpoint_intersections.iloc[heatpoint_intersections.groupby(col)[Column.DISTANCE_HEAT].idxmin()]
	return result

min_heatpoint_intersections = find_minimum_heatpoint_intersections()
min_heatpoint_intersections.to_csv("Ref_Heat.csv", index=False)
    # not neccessory old version, only to combine the data in one row
col, _ = get_main_point()
first_joined_df = pd.merge(main_df, min_h2_intersections, on=col, how="inner")
first_joined_df = pd.merge(first_joined_df, min_heatpoint_intersections, on=col, how="inner")
    # # first_joined_df.to_csv("Phase-1 Result.csv", index=False)

# Second Phase: Data management
o2_ac = pd.read_csv("O2_AC.csv", index_col=False).drop_duplicates()
ref_h2 = pd.read_csv("Ref_H2.csv", index_col=False).drop_duplicates()
ref_heat = pd.read_csv("Ref_Heat.csv", index_col=False).drop_duplicates()

# Scenario nomination for the Model 1: wwtp as refernce point 2: ac as reference point
def get_correct_ref_id_col():
	if OPTIMIZATION == "yes":
		return Column.ID_OPTIMAL

	if SCENARIO_NO == 1:
		return Column.ID_WWTP
	elif SCENARIO_NO == 2:
		return Column.ID_AC
	else:
		raise Exception("invalid ref")

def find_spec_for_ka_id(ka_id):
	found_spec = wwtp_spec[wwtp_spec[Column.ID_KA] == ka_id]

	if len(found_spec) > 1:
		raise Exception("multiple spec for a ka_id")

	found_spec = found_spec.iloc[0]

	return {
		"pe": found_spec[Column.WWTP_PE],
		"demand_o2": found_spec[Column.DEMAND_O2],
		"demand_o3": found_spec[Column.DEMAND_O3],
	}

# print("find_spec_for_ka_id: ", find_spec_for_ka_id("BB_0001"))

def get_wwtps_for_ac(ac_id):
	acs = o2_ac[o2_ac[Column.ID_AC] == ac_id]
	res = []
	for _, ac in acs.iterrows():
		res.append({
			"id": ac[Column.ID_WWTP],
			"ka_id": ac[Column.ID_KA],
			"point": from_wkt(ac[Column.POINT_WWTP]),
		})
	return res

# print("get_wwtps_for_ac: ", get_wwtps_for_ac(34056))

def get_ac_for_wwtp(wwtp_id):
	wwtp = o2_ac[o2_ac[Column.ID_WWTP] == wwtp_id]
	if len(wwtp) > 1:
		raise Exception("found multiple ac for a wwtp_id")
	wwtp = wwtp.iloc[0]
	return {
		"id": wwtp[Column.ID_AC],
		"ka_id": wwtp[Column.ID_KA],
		"point": from_wkt(wwtp[Column.POINT_AC]),
	}

# print("get_ac_for_wwtp: ", get_ac_for_wwtp(77600))

def get_heat_for_ref(ref_id):
	heat = ref_heat[ref_heat[get_correct_ref_id_col()] == ref_id]
	if len(heat) > 1:
		raise Exception("found multiple heat for a ref_id")
	heat = heat.iloc[0]
	return {
		"id": heat[Column.ID_HEAT],
		"point": from_wkt(heat[Column.POINT_HEAT]),
	}

# print("get_heat_for_ref: ", get_heat_for_ref(77600))

def get_h2_for_ref(ref_id):
	h2 = ref_h2[ref_h2[get_correct_ref_id_col()] == ref_id]
	if len(h2) > 1:
		raise Exception("found multiple h2 for a ref_id")

	h2 = h2.iloc[0]
	
	return {
		"id": h2[Column.ID_H2],
		"point": from_wkt(h2[Column.POINT_H2]),
		"type": h2[Column.TYPE_H2],
	}

def get_wwtp_point(wwpt_id):
	row = o2_ac[o2_ac[Column.ID_WWTP] == wwpt_id].iloc[0]

	return from_wkt(row[Column.POINT_WWTP])

def get_ac_point(ac_id):
	row = o2_ac[o2_ac[Column.ID_AC] == ac_id].iloc[0]

	return from_wkt(row[Column.POINT_AC])

def get_ac_distance_for_ref(ref_id, o2_to_ac):
	if OPTIMIZATION == "yes":
		row = o2_to_ac[o2_to_ac[Column.ID_OPTIMAL] == ref_id]

		if len(row) < 1:
			raise Exception("no wwtp found")

		row = row.iloc[0]

		return from_wkt(row[Column.POINT_AC]).distance(row[Column.POINT_OPTIMAL]) / 1000

	if SCENARIO_NO == 1:	
		row = o2_to_ac[o2_to_ac[Column.ID_WWTP] == ref_id]

		if len(row) != 1:
			raise Exception("multiple wwtp found")

		row = row.iloc[0]

		return from_wkt(row[Column.POINT_AC]).distance(from_wkt(row[Column.POINT_WWTP])) / 1000
	elif SCENARIO_NO == 2:
		return 0
	else:
		raise Exception("invalid scenario")
print("Intersection Completed.")
# print("get_h2_for_ref: ", get_h2_for_ref(77600))


# Second Phase: Calculation Functions
# Calculate gas pipeline diameter (O2 & H2) for further cost calculation:
def gas_pipeline_size(gas_volume_y, distance, input_pressure, molar_mass, min_pressure):
	"""
	Parameters
	----------
	gas_valume : kg/year
	distance : km
	input pressure : bar
	min pressure : bar
	molar mas : kg/mol
	Returns
    -------
    Final pressure drop [bar] & pipeline diameter [m]
	"""
	def _calculate_final_pressure(pipeline_diameter):
		flow_rate = (gas_volume_y / (8760 * molar_mass)) * UNIVERSAL_GAS_CONSTANT * TEMPERATURE / (input_pressure * 100_000)	# m3/hour
		flow_rate_s = flow_rate / 3600							# m3/second
		pipeline_area = math.pi * (pipeline_diameter / 2) ** 2	# m2
		gas_velocity = flow_rate_s / pipeline_area				# m/s
		gas_density = (input_pressure * 1e5 * molar_mass) / (UNIVERSAL_GAS_CONSTANT * TEMPERATURE)	# kg/m3
		reynolds_number = (gas_density * gas_velocity * pipeline_diameter) / UNIVERSAL_GAS_CONSTANT
		# Estimate Darcy friction factor using Moody's approximation
		darcy_friction_factor = 0.0055 * (1 + (2 * 1e4 * (2.51 / reynolds_number)) ** (1 / 3))
		# Darcy-Weisbach equation	
		pressure_drop = ((4 * darcy_friction_factor * distance * 1000 * gas_velocity ** 2) / (2 * pipeline_diameter))/1e5	# bar 
		return input_pressure - pressure_drop # bar
	for diameter in PIPELINE_DIAMETER_RANGE:
		final_pressure = _calculate_final_pressure(diameter)
		if final_pressure > min_pressure:
			return (round(final_pressure, 4), round(diameter, 4))
	raise Exception("couldn't find a final pressure < min_pressure")

# H2 pipeline diameter cost range
def get_h2_pipeline_cost(h2_pipeline_diameter):
	if h2_pipeline_diameter >= 0.5:
		return 900_000 # EUR/km
	if h2_pipeline_diameter >= 0.4:
		return 750_000 # EUR/km
	if h2_pipeline_diameter >= 0.3:
		return 600_000 # EUR/km
	if h2_pipeline_diameter >= 0.2:
		return 450_000 # EUR/km
	return 350_000 # EUR/km

# O2 pipeline diameter cost range
def get_o2_pipeline_cost(o2_pipeline_diameter):
	if o2_pipeline_diameter >= 0.5:
		return 500_000 # EUR/km
	if o2_pipeline_diameter >= 0.4:
		return 450_000 # EUR/km
	if o2_pipeline_diameter >= 0.3:
		return 400_000 # EUR/km
	if o2_pipeline_diameter >= 0.2:
		return 350_000 # EUR/km
	return 300_000 # EUR/km

# Heat cost calculation
def get_heat_pipeline_cost(p_heat_mean, heat_distance):
	if (heat_distance < 0.5) or (p_heat_mean > 5 and heat_distance < 1): # in below calculation the p_heat_mean will get 6 to model, to bypass heat capacity caculation for this phase, and later the model will multiplay the below outcome to capacity (MW) to find out the final cost of heat pipeline
		return 400_000		# [EUR/MW]
	else:
		return 400_000		# [EUR/MW]

# annualize_capital_costs [EUR/MW/YEAR or EUR/MW/KM/YEAR]
def annualize_capital_costs(overnight_costs, lifetime, p):
	"""
    Parameters
    ----------
    overnight_costs : float
        Overnight investment costs in EUR/MW or EUR/MW/km
    lifetime : int
        Number of years in which payments will be made
    p : float
        Interest rate in p.u.
    Returns
    -------
    float
        Annualized capital costs in EUR/MW/year or EUR/MW/km/year
    """
	PVA = (1 / p) - (1 / (p * (1 + p) ** lifetime)) # Present Value of Annuity
	return overnight_costs / PVA

	# Calculate WWTPs capacity base on SEC (specific energy consumption) depend on PE (population equivalent)
def calculate_wwtp_capacity(pe): # [MWh/year]
	c = "c2"
	if pe > 100_000:
		c = "c5"
	elif pe > 10_000 and pe <= 100_000:
		c = "c4"
	elif pe > 2000 and pe <= 10_000:
		c = "c3"
	return pe * WWTP_SEC[c]/1000	

# Create link between reference points and other points
def draw_lines(line_type):
	def _draw_lines(row):
		point_elz = from_wkt(row[Column.POINT_OPTIMAL])
		ac = from_wkt(row[Column.POINT_AC])
		h2 = from_wkt(row[Column.POINT_H2])
		heat = from_wkt(row[Column.POINT_HEAT])
		wwtp = from_wkt(row[Column.POINT_WWTP])
		lines = {
			"AC": LineString([[point_elz.x, point_elz.y], [ac.x, ac.y]]),			# power_to_H2
			"H2": LineString([[point_elz.x, point_elz.y], [h2.x, h2.y]]),			# H2_to_power
			"HEAT": LineString([[point_elz.x, point_elz.y], [heat.x, heat.y]]),		# power_to_heat
			"O2": LineString([[point_elz.x, point_elz.y], [wwtp.x, wwtp.y]]),		# power_to_o2
		}
		return to_wkt(lines[line_type])
	return _draw_lines


# Second Phase: Links values Calculations
# add ref_id and ref_point to o2_ac, ref_heat, ref_h2
ids = o2_ac[Column.ID_WWTP if SCENARIO_NO == 1 else Column.ID_AC].unique()
ref_ids = {id: ID_OPTIMAL_START + i for i, id in enumerate(ids)}

def add_ref_col(df):
	starting_col_id = Column.ID_WWTP if SCENARIO_NO == 1 else Column.ID_AC

	find_point = get_wwtp_point if SCENARIO_NO == 1 else get_ac_point

	df[Column.ID_OPTIMAL] = df[starting_col_id].map(ref_ids)
	df[Column.POINT_OPTIMAL] = df[starting_col_id].apply(find_point)

add_ref_col(o2_ac)
add_ref_col(ref_heat)
add_ref_col(ref_h2)

# Calculate variables for Links: power_to_O2, power_to_H2, power_to_Heat, H2_to_power
# optimized bus { "id": generated, "point": optimized }
def find_links(o2_ac, ref_heat, ref_h2):
	links = []
	total_h2_production_y = {}
	total_lcoh = 0
	found_ac = {}
	# data calculation for O2 pipeline line from ELZ to WWTP location, distance might be AC to WWTP or WWTP to WWTP (Zero)
	# future modification: 1. if distance is 0, consider it 0.1km; 2. add O2 component cost for the model
	for _, row in o2_ac.iterrows():
		if SCENARIO_NO == 2:
			if row[Column.ID_AC] in found_ac:
				continue
			else:
				found_ac[row[Column.ID_AC]] = 1

		carrier = "power_to_O2"
		if OPTIMIZATION == "yes":
			bus0 = row[Column.ID_OPTIMAL]
			bus0_point = row[Column.POINT_OPTIMAL]
			if SCENARIO_NO == 1:
				bus1s = [{"id": row[Column.ID_WWTP], "point": from_wkt(row[Column.POINT_WWTP])}]
			elif SCENARIO_NO == 2:
				bus1s = get_wwtps_for_ac(row[Column.ID_AC])
		else:
			if SCENARIO_NO == 1:
				bus0 = row[Column.ID_WWTP]
				bus0_point = from_wkt(row[Column.POINT_WWTP])
				bus1s = [{"id": bus0, "point": bus0_point}]
			elif SCENARIO_NO == 2:
				bus0 = row[Column.ID_AC]
				bus0_point = from_wkt(row[Column.POINT_AC])
				bus1s = get_wwtps_for_ac(bus0)
		
		for bus1 in bus1s:
			if SCENARIO_NO == 1:
				ka_id = row[Column.ID_KA]
			else:
				ka_id = bus1["ka_id"]
			geom = MultiLineString([[[bus0_point.x, bus0_point.y], [bus1["point"].x, bus1["point"].y]]])
			distance = bus0_point.distance(bus1["point"]) / 1000	# km
			spec = find_spec_for_ka_id(ka_id)
			wwtp_ec = calculate_wwtp_capacity(spec["pe"])			# [MWh/year]
			aeration_ec = wwtp_ec * FACTOR_AERATION_EC		# [MWh/year]
			o2_ec = aeration_ec * FACTOR_O2_EC				# [MWh/year]
			o2_ec_h = (o2_ec / 8760)						# [MWh/hour]
			total_o2_demand = (O2_O3_RATIO * spec["demand_o3"] + spec["demand_o2"] * O2_PURE_RATIO) * 1000	# kgO2/year pure O2 tonne* 1000
			h2_production_y = total_o2_demand / (O2_H2_RATIO)					# [kgH2/year]
			h2_production_h = h2_production_y / 8760
			elz_capacity = (h2_production_y * ELZ_SEC / ELZ_FLH) / 1000		# [MW]
			o2_power_ratio = o2_ec_h / elz_capacity			# will be use as constraint for the etrago model
			_,o2_pipeline_diameter = gas_pipeline_size(total_o2_demand, distance, O2_PRESSURE_ELZ, MOLAR_MASS_O2, O2_PRESSURE_MIN)

			# In below function MW shouldn't be considered since the diameter size already calcuated and KM is enough
			annualized_cost_o2_pipeline = annualize_capital_costs(get_o2_pipeline_cost(o2_pipeline_diameter), O2_LIFETIME_PIPELINE, DISCOUNT_RATE)	# [EUR/KM/YEAR]
			annualized_cost_o2_component = annualize_capital_costs(O2_COST_EQUIPMENT, O2_LIEFTIME_EQUIPMENT, DISCOUNT_RATE)	# [EUR/MW/YEAR]
			capital_cost_power_to_o2_pipeline = annualized_cost_o2_pipeline * distance # [EUR/YEAR]
			capital_cost_power_to_o2_component = annualized_cost_o2_component * o2_ec_h	# [EUR/YEAR]
			capital_cost_power_to_o2 = capital_cost_power_to_o2_pipeline + capital_cost_power_to_o2_component # 	# [EUR/YEAR]
            
			# extra explanation: the consumption energy of oxygen is already calculated in 
            #the above function (o2_ec). firstly the cost of 1kg of oxygen by considering 
            #the total consumption energy of oxygen * electricity price and totaly divide by total kg of oxygen 
            #in one year has been found, the result is the cost of 1kg of oxygen. since we already know that 
            # during production 
            # of each kg of H2 almost 8 kg (o2_h2_ratio) of O2 will be provided, the cost of 1kg of O2 will be 
            # multiplied by the h2 ratio to find the total cost of o2 in one specific time of h2 production. 
            #in further step to find the loch of o2 (by focusing on h2) the total capex of o2 pipeline
            #is required + the sellable o2 cost * h2 production per year (muliplying sellable o2 to h2 yearly
            #production give us the total contribution cost of o2 production in electrolyzer for one year) all 
            #of them will be devided to total production of hydrogen in one year to find out the total cost of 
            #EURO/kgH2. this value will be used to be sum with lcoh cost of hydrogen to show the total cost of 
            #LCOH in optimization. on the other hand it is usful to subtract this cost from sellable o2 cost to 
            #show that the cost comparison of usage of o2 by having a pipeline line. for example if the sellable
            #o2 be 0.25 and the lcoh o2 be 0.18, it show 0.07 cent benefit eventhough constructing a pipeline. 
            #if the price become negetive, it shows disadvantage of o2 usage due to high cost of pipeline.
            
			o2_selling_price = o2_ec * ELEC_COST / total_o2_demand				#EUR/kgO2
			sellable_o2 = (o2_selling_price * O2_H2_RATIO)						#EUR/kgH2
            
			lcoh_o2 = capital_cost_power_to_o2/ h2_production_y # [EUR/Year]/[kgh2/Year]   [EUR/kgH2]
			total_lcoh += lcoh_o2
			# net lcoh shows only the lcoh cost of o2 pipeline line. this might be useful to be considered if less than sellable o2 shows the advantage of using o2 pipeline line else the cost of pipeline is higher than the sellable o2 cost which shows disadvantage of using o2 pipeline line.
			etrago_cost_power_to_o2 = (annualized_cost_o2_pipeline * distance / o2_ec_h) + annualized_cost_o2_component					# [EUR/MW/YEAR]	

			links.append({
				"bus0": bus0,
				"bus1": bus1["id"],
				"carrier": carrier,
				"efficiency": O2_EFFICIENCY,
				"power_ratio": o2_power_ratio,
				"length": distance,
				"capital_cost": etrago_cost_power_to_o2,
				"lcoh_capital_cost": capital_cost_power_to_o2,
				"p_nom": o2_ec_h,
				"sellable_cost": sellable_o2,
				"LCOH": lcoh_o2,
				"elz_capacity": elz_capacity,
				"diameter":o2_pipeline_diameter,
				"ka_id": ka_id,
				"type": o2_power_ratio,
				"geom": geom,
			})
			# to accomulate H2 production demand as per O2 for the shared bus of AC
			if total_h2_production_y.get(f"{bus0}") is None:
				total_h2_production_y[f"{bus0}"] = h2_production_y
			else:
				total_h2_production_y[f"{bus0}"] += h2_production_y
	
	for _, row in ref_heat.iterrows():
		carrier = "power_to_Heat"

		if OPTIMIZATION == "yes":
			bus0 = row[Column.ID_OPTIMAL]
			bus0_point = row[Column.POINT_OPTIMAL]
		else:
			if SCENARIO_NO == 1:
				bus0 = row[Column.ID_WWTP]
				bus0_point = get_wwtp_point(bus0)
			elif SCENARIO_NO == 2:
				bus0 = row[Column.ID_AC]
				bus0_point = get_ac_point(bus0)

		bus1 = get_heat_for_ref(bus0)
		distance = bus0_point.distance(bus1["point"]) / 1000
		geom = MultiLineString([[[bus0_point.x, bus0_point.y], [bus1["point"].x, bus1["point"].y]]])

		h2_production_y = total_h2_production_y[f"{bus0}"]					# [kgH2/year]
		if h2_production_y is None:
			raise Exception("couldn't find h2_production")
		h2_production_h = h2_production_y / 8760											# [kgH2/hour]
		elz_capacity = (h2_production_y * ELZ_SEC / ELZ_FLH) / 1000							# [MW]
		heat_production_h = elz_capacity * HEAT_RATIO										# [MWh/hour]

		annualized_capex_heat = annualize_capital_costs(HEAT_COST_PIPELINE, HEAT_LIFETIME, DISCOUNT_RATE)		# EUR/MW/year
		annualized_capex_heat_pipeline = annualize_capital_costs(get_heat_pipeline_cost(6, distance),HEAT_LIFETIME, DISCOUNT_RATE)		# [EUR/MW/KM/YEAR]
		capital_cost_power_to_heat = (annualized_capex_heat + (annualized_capex_heat_pipeline * distance)) * heat_production_h			# [EUR/YEAR]

		sellable_heat = elz_capacity * HEAT_RATIO * HEAT_SELLING_PRICE / h2_production_h 	# [EUR/kgH2]
		lcoh_heat = capital_cost_power_to_heat/ h2_production_y # [EUR/kgH2]
		total_lcoh += lcoh_heat
		
		etrago_cost_power_to_heat = (annualized_capex_heat + (annualized_capex_heat_pipeline * distance))								# [EUR/MW/YEAR]

		links.append({
			"bus0": bus0,
			"bus1": bus1["id"],
			"carrier": carrier,
			"efficiency": HEAT_EFFICIENCY, 
			"power_ratio": HEAT_RATIO,
			"length": distance,
			"capital_cost":etrago_cost_power_to_heat,
			"lcoh_capital_cost": capital_cost_power_to_heat,
			"p_nom": heat_production_h,
			"sellable_cost": sellable_heat,
			"LCOH": lcoh_heat,
			"elz_capacity": elz_capacity,
			"diameter": "",
			"ka_id": HEAT_RATIO,
			"type": HEAT_RATIO,
			"geom": geom,
		})
	# data calculation for power to H2, from AC to ELZ location. distance might be AC to WWTP or AC to AC (Zero)
	# upcoming modification: 1. if distance is 0, consider it 0.1km; 2. add transformer cost; 
	# 3. since H2 pipeline is required for the project, the cost of H2 pipeline should also be considered in the capex?? ask Ulf
	for _, row in ref_h2.iterrows():
		carrier = "power_to_H2"

		if OPTIMIZATION == "yes":
			bus0 = row[Column.ID_OPTIMAL]
			bus0_point = row[Column.POINT_OPTIMAL]
		else:
			if SCENARIO_NO == 1:
				bus0 = row[Column.ID_WWTP]
				bus0_point = get_wwtp_point(bus0)
			elif SCENARIO_NO == 2:
				bus0 = row[Column.ID_AC]
				bus0_point = get_ac_point(bus0)

		bus1 = get_h2_for_ref(bus0)
		distance = bus0_point.distance(bus1["point"]) / 1000

		if SCENARIO_NO == 1:
			ac = from_wkt(o2_ac[o2_ac[Column.ID_WWTP] == row[Column.ID_WWTP]].iloc[0][Column.POINT_AC])
		elif SCENARIO_NO == 2:
			ac = get_ac_point(row[Column.ID_AC])

		geom = MultiLineString([[[bus0_point.x, bus0_point.y], [ac.x, ac.y]]])

		# Electrolyzer Calculation
		h2_production_y = total_h2_production_y[f"{bus0}"]							# [kgH2/year]
		if h2_production_y is None:
			raise Exception("couldn't find h2_production")

		h2_production_h = h2_production_y / 8760								# [kgH2/hour]
		elz_capacity = (h2_production_y * ELZ_SEC / ELZ_FLH) / 1000		# [MW]
		h2_production_energy_h = h2_production_y * 33.33 / 8760 / 1000	# [MWh/HOUR] or ELZ_capacity * ELZ_EFF
		_,h2_pipeline_diameter = gas_pipeline_size(h2_production_y, distance, H2_PRESSURE_ELZ, MOLAR_MASS_H2, H2_PRESSURE_MIN)
		ac_distance = get_ac_distance_for_ref(bus0, o2_ac)		# is this in m or km?

		# annualized cost calculation
# 		annualized_cost_h2_pipeline = annualize_capital_costs(get_h2_pipeline_cost(h2_pipeline_diameter), ELZ_LIFETIME_Y, DISCOUNT_RATE)		# [EUR/KM/YEAR]
		annualized_cost_ac_cable = annualize_capital_costs((AC_COST_CABLE * ac_distance), AC_LIFETIME_CABLE, DISCOUNT_RATE)						# [EUR/MW/YEAR]
		annualized_cost_ac_trans = annualize_capital_costs(AC_TRANS, AC_LIFETIME_CABLE, DISCOUNT_RATE)									# [EUR/MW/YEAR]
		annualized_cost_elz = annualize_capital_costs((ELZ_CAPEX_STACK + ELZ_CAPEX_SYSTEM + ELZ_OPEX), ELZ_LIFETIME_Y, DISCOUNT_RATE)			# [EUR/MW/YEAR]

		# below calcualtion aimed to find the capital cost of power to H2 for LCOH calculation for stand alone model. and the cost cover every part since the total lcoh will be the objective of the optimization in further steps.
		total_ac_cost = (annualized_cost_ac_cable + annualized_cost_ac_trans + annualized_cost_elz) * elz_capacity								# [EUR/YEAR]
# 		total_pipeline_cost = annualized_cost_h2_pipeline * distance												# [EUR/YEAR]
		lcoh_h2_elz = (total_ac_cost + (h2_production_y * ELZ_SEC * ELEC_COST/1000)) / h2_production_y	       		# [EUR/kgH2]
# 		lcoh_h2_pipeline = total_pipeline_cost / h2_production_y			# [EUR/kgH2]
		total_lcoh += lcoh_h2_elz

		# Since Capital Cost in eTraGO rquires EUR/MW/YEAR not EUR/YEAR. in addition, the power to H2 in etrago relay on cost related to produce hdyrogen and transfering the cost of H2 pipeline will be excluded and will be considered in H2 to Power link.
		etrago_annualized_cost_h2_pipeline = annualize_capital_costs(H2_COST_PIPELINE, ELZ_LIFETIME_Y, DISCOUNT_RATE) * distance	# [EUR/MW/YEAR]
		etrago_cost_power_to_h2 = annualized_cost_ac_cable + annualized_cost_ac_trans + annualized_cost_elz	+ etrago_annualized_cost_h2_pipeline	# [EUR/MW/YEAR]

		links.append({
			"bus0": bus0,
			"bus1": bus1["id"],
			"carrier": carrier,
			"efficiency": ELZ_EFF, 
			"power_ratio": ac_distance,
			"length": distance,
			"capital_cost": etrago_cost_power_to_h2,
			"lcoh_capital_cost": total_ac_cost,
			"p_nom": elz_capacity,
			"sellable_cost": "",
			"LCOH": lcoh_h2_elz,
			"elz_capacity": elz_capacity,
			"diameter": h2_pipeline_diameter,
			"ka_id": "",
			"type": bus1["type"],
			"geom": geom,
		})

	for _, row in ref_h2.iterrows():
		carrier = "H2_to_power"
		bus0 = row[Column.ID_H2]
		bus0_point = from_wkt(row[Column.POINT_H2])
		type = row[Column.TYPE_H2]
		if OPTIMIZATION == "yes":
			bus1 = row[Column.ID_OPTIMAL]
			bus1_point = row[Column.POINT_OPTIMAL]
		else:
			if SCENARIO_NO == 1:
				bus1 = row[Column.ID_WWTP]
				bus1_point = get_wwtp_point(bus1)
			elif SCENARIO_NO == 2:
				bus1 = row[Column.ID_AC]
				bus1_point = get_ac_point(bus1)
		distance = bus1_point.distance(bus0_point) / 1000
		geom = MultiLineString([[[bus1_point.x, bus1_point.y], [bus0_point.x, bus0_point.y]]])
		h2_production_y = total_h2_production_y[f"{bus1}"]					# [kgH2/year]
		h2_production_h = h2_production_y / 8760								# [kgH2/hour]
		elz_capacity = (h2_production_y * ELZ_SEC / ELZ_FLH) / 1000		# [MW]
		h2_production_energy_h = h2_production_y * 33.33 / 8760 / 1000	# [MWh/HOUR] or ELZ_capacity * ELZ_EFF

		if h2_production_y is None:
			raise Exception("couldn't find h2_production")

		_,h2_pipeline_diameter = gas_pipeline_size(h2_production_y, distance, H2_PRESSURE_ELZ, MOLAR_MASS_H2, H2_PRESSURE_MIN)

		# since the required calcualtion for this link already done in pwoer to h2 link, it is not rwquired to do it again, but only for eTraGO
		# annualized_cost_h2_pipeline = annualize_capital_costs((get_h2_pipeline_cost(h2_pipeline_diameter) * distance), ELZ_LIFETIME_Y, DISCOUNT_RATE)	# [EUR/MW]
		# capital_cost_power_to_h2 = annualized_cost_power_to_h2			# [EUR/YEAR]
		# lcoh_h2 = (capital_cost_power_to_h2 + (h2_production_y * ELZ_SEC * ELEC_COST/1000)) / h2_production_y # [EUR/kgH2]


		# calculating the cost of power to H2 for eTraGO since it is rquired EUR/MW/YEAR not EUR/YEAR
		
		annualized_cost_h2_pipeline = annualize_capital_costs(get_h2_pipeline_cost(h2_pipeline_diameter), ELZ_LIFETIME_Y, DISCOUNT_RATE)# [EUR/KM/YEAR]
        
		total_pipeline_cost = annualized_cost_h2_pipeline * distance												# [EUR/YEAR]
		lcoh_h2_pipeline = total_pipeline_cost / h2_production_y			# [EUR/kgH2]
		total_lcoh += lcoh_h2_pipeline
        
		etrago_annualized_cost_h2_pipeline = annualize_capital_costs(H2_COST_PIPELINE, ELZ_LIFETIME_Y, DISCOUNT_RATE)	# [EUR/KM/YEAR]
		etrago_cost_h2_to_power = etrago_annualized_cost_h2_pipeline * distance	# [EUR/MW/YEAR]
        
		links.append({
			"bus0": bus0,
			"bus1": bus1,
			"carrier": carrier,
			"efficiency": FUEL_CELL_EFF, 
			"power_ratio": 0,
			"length": distance,
			"capital_cost": etrago_cost_h2_to_power,
			"lcoh_capital_cost": total_pipeline_cost,
			"p_nom": h2_production_energy_h,
			"sellable_cost": "",
			"LCOH": lcoh_h2_pipeline,
			"elz_capacity": elz_capacity,
			"diameter": h2_pipeline_diameter,
			"ka_id": "",
			"type": type,
			"geom": geom,
		})
	return pd.DataFrame(links), total_lcoh


# Second Phase: Optimization function Method Nelder-Mead
unoptimized_total = 0
def find_optimal_loc(o2_ac, ref_heat, ref_h2):
	global unoptimized_total

	local_o2_ac = o2_ac.copy()
	local_ref_heat = ref_heat.copy()
	local_ref_h2 = ref_h2.copy()

	# local_o2_ac = o2_ac[o2_ac[Column.ID_OPTIMAL] <= 80010]
	# local_ref_heat = ref_heat[ref_heat[Column.ID_OPTIMAL] <= 80010]
	# local_ref_h2 = ref_h2[ref_h2[Column.ID_OPTIMAL] <= 80010]

	links_df, unoptimized_total = find_links(local_o2_ac, local_ref_heat, local_ref_h2)

	# print("Unoptimized total LCOH=", unoptimized_total)

	# links_df.to_csv(f'SCN-{SCENARIO_NO} Before Optimization.csv', index=False)

	# filter H2_to_power links
	filtered = links_df[links_df["carrier"] != "H2_to_power"]

	unique_optimal_ids = filtered["bus0"].unique()

	# print("unique_optimal_ids_count=", len(unique_optimal_ids))

	for id in unique_optimal_ids:
		filtered_o2_ac = local_o2_ac[local_o2_ac[Column.ID_OPTIMAL] == id]
		filtered_ref_heat = local_ref_heat[local_ref_heat[Column.ID_OPTIMAL] == id]
		filtered_ref_h2 = local_ref_h2[local_ref_h2[Column.ID_OPTIMAL] == id]

		def _total_cost(center):
			filtered_o2_ac.loc[filtered_o2_ac[Column.ID_OPTIMAL] == id, Column.POINT_OPTIMAL] = Point(center)
			filtered_ref_h2.loc[filtered_ref_h2[Column.ID_OPTIMAL] == id, Column.POINT_OPTIMAL] = Point(center)
			filtered_ref_heat.loc[filtered_ref_heat[Column.ID_OPTIMAL] == id, Column.POINT_OPTIMAL] = Point(center)

			try:
				_, lcoh = find_links(filtered_o2_ac, filtered_ref_heat, filtered_ref_h2)
			except:
				return math.inf

			return lcoh

		x = filtered_o2_ac[Column.POINT_OPTIMAL].iloc[0].x
		y = filtered_o2_ac[Column.POINT_OPTIMAL].iloc[0].y
		optimal_point = minimize(_total_cost, [x, y], method = 'Nelder-Mead')

		local_o2_ac.loc[local_o2_ac[Column.ID_OPTIMAL] == id, Column.POINT_OPTIMAL] = (Point(optimal_point.x))
		local_ref_heat.loc[local_ref_heat[Column.ID_OPTIMAL] == id, Column.POINT_OPTIMAL] = (Point(optimal_point.x))
		local_ref_h2.loc[local_ref_h2[Column.ID_OPTIMAL] == id, Column.POINT_OPTIMAL] = (Point(optimal_point.x))
	
	return local_o2_ac, local_ref_heat, local_ref_h2

# Second Phase: running the optimization
if OPTIMIZATION == "yes":
	a, b, c = find_optimal_loc(o2_ac, ref_heat, ref_h2)
	links_df, optimized_total = find_links(a, b, c)
	print("optimized total LCOH: ", optimized_total)
	print("diff: ", unoptimized_total - optimized_total)
	links_df.to_csv(f'SCN-{SCENARIO_NO} Optimized.csv', index=False)
else:
	links_df, _ = find_links(o2_ac, ref_heat, ref_h2)
	links_df.to_csv(f'SCN-{SCENARIO_NO} Original.csv', index=False)
	print("Optimization is not selected")


# Third Phase: Export to PostgreSQL
# export links data to PostgreSQL database
if OPTIMIZATION == "no":
	def export_to_db(df):
		df = df.copy(deep=True)
		etrago_columns = [	
			"scn_name",
			"link_id",
			"bus0",
			"bus1",
			"carrier",
			"efficiency",
			"build_year",
			"lifetime",
			"p_nom",
			"p_nom_extendable",
			"capital_cost",
			"length",
			"terrain_factor",
            "type",
			"geom",
		]
		max_link_id = 80_000
		next_max_link_id = count(start=max_link_id, step=1)

		df["scn_name"] = SCENARIO_NAME
		df["build_year"] = 2035
		df["lifetime"] = 25
		df["p_nom_extendable"] = True
		df["length"] = 0
		df["link_id"] = df["bus0"].apply(lambda _: next(next_max_link_id))
		df["geom"] = df["geom"].apply(lambda x: to_wkt(x))

		df = df.filter(items=etrago_columns, axis=1)

		table_name = "egon_etrago_link"
		with engine.connect() as conn:
			conn.execute(text(f"DELETE FROM grid.{table_name} WHERE link_id >= {max_link_id} AND scn_name = '{SCENARIO_NAME}'"))

		df.to_sql(
			"egon_etrago_link",
			engine,
			schema="grid",
			if_exists="append",
			index=False
		)
	print("link data exported to: egon_etrago_link")
	export_to_db(links_df)
else:
	print("Optimized, but link data has not been imported to PostgreSQL")
	

# Third Phase: Export O2 load to PostgreSQL
if OPTIMIZATION == "no":
	def insert_load_points(df):
		max_load_id = 40_000
		next_load_id = count(start=max_load_id, step=1)
		table_name = "egon_etrago_load"

		with engine.connect() as conn:
			conn.execute(f"DELETE FROM grid.{table_name} WHERE load_id >= {max_load_id} AND scn_name = '{SCENARIO_NAME}'")
		
		df = df.copy(deep=True)

		df = df[df["carrier"] == "power_to_O2"]

		result = []
		for _, row in df.iterrows():
			load_id = next(next_load_id)

			result.append({
				"scn_name": SCENARIO_NAME,
				"load_id": load_id,
				"bus": row["bus1"], 
				"carrier": "O2",
				"type" : "O2",
				"p_set": row["p_nom"],

			})
		df = pd.DataFrame(result)

		df.to_sql(
			table_name,
			engine,
			schema="grid",
			if_exists="append",
			index=False
		)
	print("load data exported to: egon_etrago_load")

	insert_load_points(links_df)
else:
	print("Optimized, but load data has not been imported to PostgreSQL")
	

# Third Phase: Export O2 generator to O2 bus points in to the PostgreSQL database
if OPTIMIZATION == "no":
	def insert_generator_points(df):
		max_generator_id = 40_000
		next_generator_id = count(start=max_generator_id, step=1)
		table_name = "egon_etrago_generator"

		with engine.connect() as conn:
			conn.execute(f"DELETE FROM grid.{table_name} WHERE generator_id >= {max_generator_id} AND scn_name = '{SCENARIO_NAME}'")

		df = df.copy(deep=True)

		df = df[df["carrier"] == "power_to_O2"]

		result = []
		for _, row in df.iterrows():
			generator_id = next(next_generator_id)

			result.append({
				"scn_name": SCENARIO_NAME,
				"generator_id": generator_id,
				"bus": row["bus1"],
				"carrier": "O2",
				"p_nom_extendable": 'true',
				"type": 'O2',
				"marginal_cost": ELEC_COST, #ELEC_COST, # row[Column.O2_H2_SELL],
			})
		df = pd.DataFrame(result)

		df.to_sql(
			table_name,
			engine,
			schema="grid",
			if_exists="append",
			index=False
		)
	print("generator data exported to: egon_etrago_generator")
	insert_generator_points(links_df)
else:
	print("Optimized, but generator data has not been imported to PostgreSQL")

## Third Phase: Export load time series data to PostgreSQL database
# if OPTIMIZATION == "no":
# 	def insert_load_timeseries(df):
# 		max_loadt_id = 40_000
# 		next_loadt_id = count(start=max_loadt_id, step=1)
# 		table_name = "egon_etrago_load_timeseries"
		
# 		with engine.connect() as conn:
# 			conn.execute(f"DELETE FROM grid.{table_name} WHERE load_id >= {max_loadt_id} AND scn_name = '{SCENARIO_NAME}'")

# 		df = df.copy(deep=True)

# 		df = df[df["carrier"] == "power_to_O2"]
		
# 		result = []
# 		for _, row in df.iterrows():
# 			load_id = next(next_loadt_id)
			
# 			# Repeat the p_set_value 8760 times  
# 			p_set_value = row["p_nom"]  # Assuming this is the single value you have
# 			p_value = row["p_nom"]
# 			hours = 8760
# 			fluctuation = 0.3  # 30%
# 			p_random_values = [p_value * (1 + random.uniform(-fluctuation, fluctuation)) for _ in range(hours)]
# 			current_sum = sum(p_random_values)
# 			desired_sum = p_value * hours
# 			scaling_factor = desired_sum / current_sum
# 			adjusted_p_value = [value * scaling_factor for value in p_random_values]
			
# 			result.append({
# 				"scn_name": SCENARIO_NAME,
# 				"load_id": load_id,
# 				"temp_id": 1,
# 				"p_set": adjusted_p_value
# 			})        
# 		df = pd.DataFrame(result)
		
# 		df.to_sql(
# 			table_name,
# 			engine,
# 			schema="grid",
# 			if_exists="append",
# 			index=False,
# 			# dtype= {"p_set": "DOUBLE PRECISION[]"}
# 		)
#	print("load timeseries data has been imported to PostgreSQL")
# 	insert_load_timeseries(links_df)
# else:
#	print("Optimized, but load timeseries data has not been imported to PostgreSQL")

