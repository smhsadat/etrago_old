import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point
from shapely.wkb import dumps
from shapely.io import from_wkt, to_wkb

# Database connection settings
engine = create_engine(
    "postgresql+psycopg2://postgres:"
    "postgres@localhost:"
    "5432/etrago-data",
    echo=False)

# Read the CSV file containing the new data
points_df = pd.read_csv("WWTP_to_postgresql.csv")

# Convert coordinates to Shapely geometries
points_df['geom'] = points_df.apply(lambda row: dumps(Point(row['x'], row['y']), srid=4326), axis=1)

# Save the DataFrame to a new CSV file
points_df.to_csv("WWTP_to_postgresql.csv", index=False)

# Create a database engine and write the DataFrame to a PostgreSQL table
points_df.to_sql(
    "egon_etrago_bus",
    engine,
    schema="grid",
    if_exists="append",
    index=False
)

print("Operation completed")

# # Dump H2 Grid into database
# first_phase_output = pd.read_csv("D:\Thesis\Model & Data\Scenario1 Multiline_loc.csv")

# def create_h2_grid_point(bus_id, geom):
#     point = from_wkt(geom)

#     return {
#         "scn_name": "eGon2035",
#         "bus_id": bus_id,
#         "type": "",
#         "carrier": "h2_grid",
#         "x": point.x,
#         "y": point.y,
#         "geom": to_wkb(point),
#     }

# h2_grid_points = []

# for _, row in first_phase_output:
#     h2_id = row["H2_ID"]
#     geom = row["POINT_H2"]

#     if row["TYPE_H2"] == "h2_grid":
#         grid_point = create_h2_grid_point(h2_id, geom)
#         h2_grid_points.append(grid_point)
    
# h2_grid_point_df = pd.DataFrame(h2_grid_points)

# h2_grid_points.head()