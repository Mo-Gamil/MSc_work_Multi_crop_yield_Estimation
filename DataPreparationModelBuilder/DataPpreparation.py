# -*- coding: utf-8 -*-
"""
Generated by ArcGIS ModelBuilder on : 2024-06-24 23:05:11
"""
import arcpy

def DataPreparation():  # DataPpreparation

    # To allow overwriting outputs change overwriteOutput option to True.
    arcpy.env.overwriteOutput = False

    arcpy.ImportToolbox(r"c:\program files\arcgis\pro\Resources\ArcToolbox\toolboxes\Data Management Tools.tbx")
    arcpy.ImportToolbox(r"c:\program files\arcgis\pro\Resources\ArcToolbox\toolboxes\Conversion Tools.tbx")
    arcpy.ImportToolbox(r"C:\Users\mhale\OneDrive\Documents\ArcGIS\Projects\Thesis_work\SplitByStateTools.atbx")
    USA_State_4_ = "USA_State"
    USA_State = "USA_State"
    dtl_cnty = "dtl_cnty"
    random_points_filtered = "random_points_filtered"
    Thesis_work_gdb = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb"
    South_Dakota = "South_Dakota"
    North_Dakota = "North_Dakota"
    Minnesota = "Minnesota"
    Kansas = "Kansas"
    Iowa = "Iowa"
    Indiana = "Indiana"
    Illinois = "Illinois"
    South_Dakota_2_ = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\South_Dakota"
    South_Dakota_32614_5_ = "D:\\Thesis Work\\Data\\corrected_shapefiles\\South_Dakota_32614"
    North_Dakota_2_ = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\North_Dakota"
    North_Dakota_32614_5_ = "D:\\Thesis Work\\Data\\corrected_shapefiles\\North_Dakota_32614"
    Minnesota_2_ = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Minnesota"
    Minnesota_32615_5_ = "D:\\Thesis Work\\Data\\corrected_shapefiles\\Minnesota_32615"
    Kansas_2_ = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Kansas"
    Kansas_32615_5_ = "D:\\Thesis Work\\Data\\corrected_shapefiles\\Kansas_32615"
    Iowa_2_ = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Iowa"
    Iowa_32615_5_ = "D:\\Thesis Work\\Data\\corrected_shapefiles\\Iowa_32615"
    Indiana_2_ = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Indiana"
    Indiana_32615_5_ = "D:\\Thesis Work\\Data\\corrected_shapefiles\\Indiana_32615"
    Illinois_2_ = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Illinois"
    Illinois_32616_5_ = "D:\\Thesis Work\\Data\\corrected_shapefiles\\Illinois_32616"
    shp_2_ = "D:\\Thesis Work\\Data\\CDL\\South_Dakota\\shp"
    shp_4_ = "D:\\Thesis Work\\Data\\CDL\\North_Dakota\\shp"
    shp_6_ = "D:\\Thesis Work\\Data\\CDL\\Minnesota\\shp"
    shp_8_ = "D:\\Thesis Work\\Data\\CDL\\Kansas\\shp"
    shp_10_ = "D:\\Thesis Work\\Data\\CDL\\Iowa\\shp"
    shp_12_ = "D:\\Thesis Work\\Data\\CDL\\Indiana\\shp"
    shp_14_ = "D:\\Thesis Work\\Data\\CDL\\Illinois\\shp"

    # Process: Select Layer By Attribute (Select Layer By Attribute) (management)
    USA_State_2_, Count = arcpy.management.SelectLayerByAttribute(in_layer_or_view=USA_State_4_, where_clause="STATE_NAME = 'Illinois' Or STATE_NAME = 'Iowa' Or STATE_NAME = 'Kansas' Or STATE_NAME = 'North Dakota' Or STATE_NAME = 'South Dakota' Or STATE_NAME = 'Minnesota' Or STATE_NAME = 'Indiana'")

    # Process: Select Layer By Location (Select Layer By Location) (management)
    dtl_cnty_2_, Output_Layer_Names, Count_2_ = arcpy.management.SelectLayerByLocation(in_layer=[dtl_cnty], overlap_type="INTERSECT", select_features=USA_State_2_)

    # Process: Export Features (Export Features) (conversion)
    counties_of_interest = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\counties_of_interest"
    arcpy.conversion.ExportFeatures(in_features=dtl_cnty_2_, out_features=counties_of_interest, field_mapping="NAME \"NAME\" true true false 50 Text 0 0,First,#,dtl_cnty,NAME,0,50;STATE_NAME \"STATE_NAME\" true true false 20 Text 0 0,First,#,dtl_cnty,STATE_NAME,0,20;STATE_ABBR \"STATE_ABBR\" true true false 2 Text 0 0,First,#,dtl_cnty,STATE_ABBR,0,2;STATE_FIPS \"STATE_FIPS\" true true false 2 Text 0 0,First,#,dtl_cnty,STATE_FIPS,0,2;COUNTY_FIPS \"COUNTY_FIPS\" true true false 3 Text 0 0,First,#,dtl_cnty,COUNTY_FIPS,0,3;FIPS \"FIPS\" true true false 5 Text 0 0,First,#,dtl_cnty,FIPS,0,5;POPULATION \"POPULATION_2020\" true true false 0 Long 0 0,First,#,dtl_cnty,POPULATION,-1,-1;POP_SQMI \"POP20_SQMI\" true true false 0 Double 0 0,First,#,dtl_cnty,POP_SQMI,-1,-1;SQMI \"SQMI\" true true false 0 Double 0 0,First,#,dtl_cnty,SQMI,-1,-1;Shape__Area \"Shape__Area\" false true true 0 Double 0 0,First,#,dtl_cnty,Shape__Area,-1,-1;Shape__Length \"Shape__Length\" false true true 0 Double 0 0,First,#,dtl_cnty,Shape__Length,-1,-1")

    # Process: Intersect (Intersect) (analysis)
    points_rand_inside_counties = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\points_rand_inside_counties"
    arcpy.analysis.Intersect(in_features=[[counties_of_interest, ""], [random_points_filtered, ""]], out_feature_class=points_rand_inside_counties)

    # Process: Add Field (2) (Add Field) (management)
    points_rand_inside_counties_5_ = arcpy.management.AddField(in_table=points_rand_inside_counties, field_name="long", field_type="DOUBLE", field_alias="long")[0]

    # Process: Add Field (Add Field) (management)
    points_rand_inside_counties_4_ = arcpy.management.AddField(in_table=points_rand_inside_counties, field_name="lat", field_type="DOUBLE", field_alias="lat")[0]

    # Process: Calculate Geometry Attributes (Calculate Geometry Attributes) (management)
    points_rand_Intersect_2_ = arcpy.management.CalculateGeometryAttributes(in_features=points_rand_inside_counties_4_, geometry_property=[["long", "POINT_X"], ["lat", "POINT_Y"]])[0]

    # Process: Add Field (3) (Add Field) (management)
    points_rand_inside_counties_3_ = arcpy.management.AddField(in_table=points_rand_inside_counties, field_name="county_name", field_type="TEXT", field_alias="county_name")[0]

    # Process: Calculate Field (Calculate Field) (management)
    points_rand_inside_counties_2_ = arcpy.management.CalculateField(in_table=points_rand_inside_counties_3_, field="county_name", expression="!NAME!.split()[0].upper()")[0]

    # Process: Export Features (2) (Export Features) (conversion)
    USA_State_of_interest = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\USA_State_of_interest"
    arcpy.conversion.ExportFeatures(in_features=USA_State_2_, out_features=USA_State_of_interest, field_mapping="STATE_ABBR \"State Abbreviation\" true true false 2 Text 0 0,First,#,USA_State,STATE_ABBR,0,2;STATE_FIPS \"State FIPS\" true true false 2 Text 0 0,First,#,USA_State,STATE_FIPS,0,2;STATE_NAME \"State Name\" true true false 25 Text 0 0,First,#,USA_State,STATE_NAME,0,25;POPULATION \"2022 Total Population\" true true false 0 Long 0 0,First,#,USA_State,POPULATION,-1,-1;POP_SQMI \"2022 Population per square mile\" true true false 0 Double 0 0,First,#,USA_State,POP_SQMI,-1,-1;SQMI \"Area in square miles\" true true false 0 Double 0 0,First,#,USA_State,SQMI,-1,-1;POPULATION_2020 \"2020 Total Population\" true true false 0 Long 0 0,First,#,USA_State,POPULATION_2020,-1,-1;POP20_SQMI \"2020 Population per square mile\" true true false 0 Double 0 0,First,#,USA_State,POP20_SQMI,-1,-1;Shape__Area \"Shape__Area\" false true true 0 Double 0 0,First,#,USA_State,Shape__Area,-1,-1;Shape__Length \"Shape__Length\" false true true 0 Double 0 0,First,#,USA_State,Shape__Length,-1,-1")

    # Process: Create Random Points (Create Random Points) (management)
    random_points = arcpy.management.CreateRandomPoints(out_path=Thesis_work_gdb, out_name="random_points", constraining_feature_class=USA_State_of_interest, number_of_points_or_field=50000, minimum_allowed_distance="6000 Meters")[0]

    # Process: Project (6) (Project) (management)
    South_Dakota_32614 = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\South_Dakota_32614"
    arcpy.management.Project(in_dataset=South_Dakota_2_, out_dataset=South_Dakota_32614, out_coor_system="PROJCS[\"WGS_1984_UTM_Zone_14N\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",500000.0],PARAMETER[\"False_Northing\",0.0],PARAMETER[\"Central_Meridian\",-99.0],PARAMETER[\"Scale_Factor\",0.9996],PARAMETER[\"Latitude_Of_Origin\",0.0],UNIT[\"Meter\",1.0]]")

    # Process: Feature Class To Shapefile (Feature Class To Shapefile) (conversion)
    South_Dakota_32614_4_ = arcpy.conversion.FeatureClassToShapefile(Input_Features=[South_Dakota_32614], Output_Folder=South_Dakota_32614_5_)[0]

    # Process: Project (Project) (management)
    North_Dakota_32614 = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\North_Dakota_32614"
    arcpy.management.Project(in_dataset=North_Dakota_2_, out_dataset=North_Dakota_32614, out_coor_system="PROJCS[\"WGS_1984_UTM_Zone_14N\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",500000.0],PARAMETER[\"False_Northing\",0.0],PARAMETER[\"Central_Meridian\",-99.0],PARAMETER[\"Scale_Factor\",0.9996],PARAMETER[\"Latitude_Of_Origin\",0.0],UNIT[\"Meter\",1.0]]")

    # Process: Add Field (11) (Add Field) (management)
    North_Dakota_32614_2_ = arcpy.management.AddField(in_table=North_Dakota_32614, field_name="x_cord", field_type="LONG", field_alias="x_cord")[0]

    # Process: Feature Class To Shapefile (2) (Feature Class To Shapefile) (conversion)
    North_Dakota_32614_4_ = arcpy.conversion.FeatureClassToShapefile(Input_Features=[North_Dakota_32614_2_], Output_Folder=North_Dakota_32614_5_)[0]

    # Process: Project (2) (Project) (management)
    Minnesota_32615 = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Minnesota_32615"
    arcpy.management.Project(in_dataset=Minnesota_2_, out_dataset=Minnesota_32615, out_coor_system="PROJCS[\"WGS_1984_UTM_Zone_15N\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",500000.0],PARAMETER[\"False_Northing\",0.0],PARAMETER[\"Central_Meridian\",-93.0],PARAMETER[\"Scale_Factor\",0.9996],PARAMETER[\"Latitude_Of_Origin\",0.0],UNIT[\"Meter\",1.0]]")

    # Process: Add Field (10) (Add Field) (management)
    Minnesota_32615_2_ = arcpy.management.AddField(in_table=Minnesota_32615, field_name="x_cord", field_type="LONG", field_alias="x_cord")[0]

    # Process: Feature Class To Shapefile (3) (Feature Class To Shapefile) (conversion)
    Minnesota_32615_4_ = arcpy.conversion.FeatureClassToShapefile(Input_Features=[Minnesota_32615_2_], Output_Folder=Minnesota_32615_5_)[0]

    # Process: Project (3) (Project) (management)
    Kansas_32615 = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Kansas_32615"
    arcpy.management.Project(in_dataset=Kansas_2_, out_dataset=Kansas_32615, out_coor_system="PROJCS[\"WGS_1984_UTM_Zone_15N\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",500000.0],PARAMETER[\"False_Northing\",0.0],PARAMETER[\"Central_Meridian\",-93.0],PARAMETER[\"Scale_Factor\",0.9996],PARAMETER[\"Latitude_Of_Origin\",0.0],UNIT[\"Meter\",1.0]]")

    # Process: Add Field (9) (Add Field) (management)
    Kansas_32615_2_ = arcpy.management.AddField(in_table=Kansas_32615, field_name="x_cord", field_type="LONG", field_alias="x_cord")[0]

    # Process: Feature Class To Shapefile (4) (Feature Class To Shapefile) (conversion)
    Kansas_32615_4_ = arcpy.conversion.FeatureClassToShapefile(Input_Features=[Kansas_32615_2_], Output_Folder=Kansas_32615_5_)[0]

    # Process: Project (4) (Project) (management)
    Iowa_32615 = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Iowa_32615"
    arcpy.management.Project(in_dataset=Iowa_2_, out_dataset=Iowa_32615, out_coor_system="PROJCS[\"WGS_1984_UTM_Zone_15N\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",500000.0],PARAMETER[\"False_Northing\",0.0],PARAMETER[\"Central_Meridian\",-93.0],PARAMETER[\"Scale_Factor\",0.9996],PARAMETER[\"Latitude_Of_Origin\",0.0],UNIT[\"Meter\",1.0]]")

    # Process: Add Field (8) (Add Field) (management)
    Iowa_32615_2_ = arcpy.management.AddField(in_table=Iowa_32615, field_name="x_cord", field_type="LONG", field_alias="x_cord")[0]

    # Process: Feature Class To Shapefile (5) (Feature Class To Shapefile) (conversion)
    Iowa_32615_4_ = arcpy.conversion.FeatureClassToShapefile(Input_Features=[Iowa_32615_2_], Output_Folder=Iowa_32615_5_)[0]

    # Process: Project (5) (Project) (management)
    Indiana_32615 = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Indiana_32615"
    arcpy.management.Project(in_dataset=Indiana_2_, out_dataset=Indiana_32615, out_coor_system="PROJCS[\"WGS_1984_UTM_Zone_16N\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",500000.0],PARAMETER[\"False_Northing\",0.0],PARAMETER[\"Central_Meridian\",-87.0],PARAMETER[\"Scale_Factor\",0.9996],PARAMETER[\"Latitude_Of_Origin\",0.0],UNIT[\"Meter\",1.0]]")

    # Process: Add Field (7) (Add Field) (management)
    Indiana_32615_2_ = arcpy.management.AddField(in_table=Indiana_32615, field_name="x_cord", field_type="LONG", field_alias="x_cord")[0]

    # Process: Feature Class To Shapefile (6) (Feature Class To Shapefile) (conversion)
    Indiana_32615_4_ = arcpy.conversion.FeatureClassToShapefile(Input_Features=[Indiana_32615_2_], Output_Folder=Indiana_32615_5_)[0]

    # Process: Project (7) (Project) (management)
    Illinois_32616 = "C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb\\Illinois_32616"
    arcpy.management.Project(in_dataset=Illinois_2_, out_dataset=Illinois_32616, out_coor_system="PROJCS[\"WGS_1984_UTM_Zone_16N\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"False_Easting\",500000.0],PARAMETER[\"False_Northing\",0.0],PARAMETER[\"Central_Meridian\",-87.0],PARAMETER[\"Scale_Factor\",0.9996],PARAMETER[\"Latitude_Of_Origin\",0.0],UNIT[\"Meter\",1.0]]")

    # Process: Add Field (6) (Add Field) (management)
    Illinois_32616_2_ = arcpy.management.AddField(in_table=Illinois_32616, field_name="x_cord", field_type="LONG", field_alias="x_cord")[0]

    # Process: Feature Class To Shapefile (7) (Feature Class To Shapefile) (conversion)
    Illinois_32616_4_ = arcpy.conversion.FeatureClassToShapefile(Input_Features=[Illinois_32616_2_], Output_Folder=Illinois_32616_5_)[0]

    # Process: Add Field (4) (Add Field) (management)
    South_Dakota_32614_2_ = arcpy.management.AddField(in_table=South_Dakota_32614, field_name="x_cord", field_type="LONG", field_alias="x_cord")[0]

    # Process: Calculate Field (2) (Calculate Field) (management)
    South_Dakota_32614_3_ = arcpy.management.CalculateField(in_table=South_Dakota_32614_2_, field="x_cord", expression="!OBJECTID!")[0]

    # Process: Calculate Field (3) (Calculate Field) (management)
    North_Dakota_32614_3_ = arcpy.management.CalculateField(in_table=North_Dakota_32614_2_, field="x_cord", expression="!OBJECTID!")[0]

    # Process: Calculate Field (4) (Calculate Field) (management)
    Minnesota_32615_3_ = arcpy.management.CalculateField(in_table=Minnesota_32615_2_, field="x_cord", expression="!OBJECTID!")[0]

    # Process: Calculate Field (5) (Calculate Field) (management)
    Kansas_32615_3_ = arcpy.management.CalculateField(in_table=Kansas_32615_2_, field="x_cord", expression="!OBJECTID!")[0]

    # Process: Calculate Field (6) (Calculate Field) (management)
    Iowa_32615_3_ = arcpy.management.CalculateField(in_table=Iowa_32615_2_, field="x_cord", expression="!OBJECTID!")[0]

    # Process: Calculate Field (7) (Calculate Field) (management)
    Indiana_32615_3_ = arcpy.management.CalculateField(in_table=Indiana_32615_2_, field="x_cord", expression="!OBJECTID!")[0]

    # Process: Calculate Field (8) (Calculate Field) (management)
    Illinois_32616_3_ = arcpy.management.CalculateField(in_table=Illinois_32616_2_, field="x_cord", expression="!OBJECTID!")[0]

    # Process: ScriptScriptSplitByStateTools (ScriptScriptSplitByStateTools) (SplitByStateTools)
    arcpy.SplitByStateTools.ScriptSplitByStateTools(input_fc=points_rand_inside_counties, output_workspace=Thesis_work_gdb)

if __name__ == '__main__':
    # Global Environment settings
    with arcpy.EnvManager(scratchWorkspace="C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb", workspace="C:\\Users\\mhale\\OneDrive\\Documents\\ArcGIS\\Projects\\Thesis_work\\Thesis_work.gdb"):
        DataPreparation()
