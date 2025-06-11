import psycopg2
import geopandas as gpd
from sqlalchemy import create_engine, String, Float, MetaData, Table, select, distinct
import re

def create_database(name = "ctroads"):
    pg_conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        dbname="postgres"   # connect to default db to run CREATE DATABASE
    )
    pg_conn.autocommit = True
    with pg_conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE {name}")
    pg_conn.close()

def create_roads_table():
    # 1. Read shapefile
    gdf = gpd.read_file("ct_roads/ct_roads_2024.shp")

    # 2. First reproject to a projected CRS (UTM zone for Connecticut)
    # Connecticut is in UTM zone 18N
    gdf = gdf.to_crs(epsg=32618)  # UTM zone 18N
    
    # 3. Compute centroids (now in a projected CRS for accurate results)
    gdf["centroid"] = gdf.geometry.centroid
    
    # 4. Reproject back to WGS84 for lon/lat coordinates
    gdf = gdf.to_crs(epsg=4326)
    gdf["lon"] = gdf.centroid.x
    gdf["lat"] = gdf.centroid.y

    # 5. Apply normalization to street names
    gdf["NORMSTREETNAME"] = gdf["FULLNAME"].apply(normalize_street_name)

    # 6. Select the columns you want
    roads_df = gdf[["LINEARID", "FULLNAME", "NORMSTREETNAME", "lon", "lat"]].copy()
    return roads_df

def write_roads_table_to_db(df, db = "ctroads"):
    engine = create_engine(
        f"postgresql+psycopg2://postgres:5342@localhost:5432/{db}"
    )

    # Write to a table named "roads"; replace if it already exists
    df.to_sql(
        "roads",
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "LINEARID": String,  
            "FULLNAME": String,
            "NORMSTREETNAME": String,  
            "lon": Float,     
            "lat": Float     
        }
    )
    print("Done: roads table created in database 'ctroads'.")

def query_roads_table(db = "ctroads"):
    """
    For testing purposes - to make sure the database is working, print all the roads
    below latitude 41.2.
    """
    # 1. Create your engine (replace with your actual credentials)
    engine = create_engine(
        f"postgresql+psycopg2://postgres:5342@localhost:5432/{db}"
    )

    # 2. Reflect the 'roads' table
    metadata = MetaData()
    roads = Table("roads", metadata, autoload_with=engine)

    # 3. Build the select statement
    stmt = (
        select(distinct(roads.c.NORMSTREETNAME))
        .where(roads.c.lat < 41.2)
        .order_by(roads.c.NORMSTREETNAME)
    )

    # 4. Execute and fetch
    with engine.connect() as conn:
        result = conn.execute(stmt)
        road_names = [row[0] for row in result]

    # 5. Do something with the list (here, we just print them)
    for name in road_names:
        print(name)

def normalize_street_name(raw: str) -> str:
    """
    Normalize a street name:
      - uppercase
      - remove punctuation
      - expand directions & suffixes
      - collapse whitespace
      - title-case output
    """
    _SUFFIXES = {
        r'\bST\b':      'STREET',
        r'\bRD\b':      'ROAD',
        r'\bAVE\b':     'AVENUE',
        r'\bBLVD\b':    'BOULEVARD',
        r'\bDR\b':      'DRIVE',
        r'\bCT\b':      'COURT',
        r'\bLN\b':      'LANE',
        r'\bPL\b':      'PLACE',
        r'\bCIR\b':     'CIRCLE',
        r'\bTRL\b':     'TRAIL',
        r'\bPKWY\b':    'PARKWAY',
        r'\bHWY\b':     'HIGHWAY',
    }

    _DIRECTIONS = {
        r'\bN\b':       'NORTH',
        r'\bS\b':       'SOUTH',
        r'\bE\b':       'EAST',
        r'\bW\b':       'WEST',
        r'\bNE\b':      'NORTHEAST',
        r'\bNW\b':      'NORTHWEST',
        r'\bSE\b':      'SOUTHEAST',
        r'\bSW\b':      'SOUTHWEST',
    }

    if not raw:
        return ''

    s = raw.strip().upper()
    # replace any punctuation (except underscores) with space
    s = re.sub(r'[^\w\s]', ' ', s)

    # expand directions first (so 'N Main St' -> 'North Main St')
    for pat, full in _DIRECTIONS.items():
        s = re.sub(pat, full, s)

    # then expand suffixes
    for pat, full in _SUFFIXES.items():
        s = re.sub(pat, full, s)

    # collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()

    # title-case for readability
    return s.title()

if __name__ == "__main__":
    query_roads_table()