using DataFrames
using CSV
import Statistics as stat
cd(joinpath(@__DIR__, "data"))
starting_year = 2013
# I put Alberta as 9999 in the dataset, so that it is not considered as a country
Trade_data = DataFrame(CSV.File("1- CEPII_Processed_HS4_$(starting_year)_2023.csv"));
#Trade_data.exporter .= ifelse.(Trade_data.exporter .== "Alberta", "9999", Trade_data.exporter);
#Trade_data.exporter = parse.(Int64, Trade_data.exporter);


#**********
Country_list = DataFrame(CSV.File("country_list.csv"));
# Adding Alberta to the Country_list
new_row = (Country = "Alberta", ISO_3 = "AB", PartnerCode= 9999)  
push!(Country_list, new_row)
println("Total Number of Transactions [$(starting_year), 2023] at HS4 level: $(nrow(Trade_data))")
filter!(row -> (row.exporter in Country_list.PartnerCode) & (row.importer in Country_list.PartnerCode), Trade_data);
println("Number of Transactions among Alberta's top 30 trading partner: $(nrow(Trade_data))")

#**********
# For some countries (e.g., US), we have multiple country codes. 
# Here I ensure that we only keep one code for each country.
# then all the datasets that are gonna be merged with Trade_data I will ensure to use the same code.
Country_list
# Belgium: Keep 58, remove 56
# Switzerland: Keep 757, remove 756
# France: Keep 251, remove 250
# India: Keep 699, remove 356
# United States: Keep 842, remove 841 and 840
# Vietnam: Keep 868, remove 704
exporter_mapping = Dict(
    56 => 58,
    756 => 757,
    250 => 251,
    356 => 699,
    841 => 842,
    840 => 842,
    704 => 868
);
target_values = [58, 757, 251, 699, 842, 868] ;

# mapping importer and exporter codes to the ones in the Country_list
Trade_data.importer = map(x -> get(exporter_mapping, x, x), Trade_data.importer);
Trade_data.exporter = map(x -> get(exporter_mapping, x, x), Trade_data.exporter);


#**********
Commodity_Policies = DataFrame(CSV.File("country_cmd_policy.csv"));
describe(Commodity_Policies)
rename!(Commodity_Policies, "yrs" => "year");
rename!(Commodity_Policies, :HS4 => :hsCode);
# mapping exporter codes to the ones in the Country_list
    # 1- ensuring that all the target values are in the dataset 
missing_values = setdiff(target_values, unique(Commodity_Policies.importer))
println("Missing values: ", missing_values)
missing_values = setdiff(target_values, unique(Commodity_Policies.exporter))
println("Missing values: ", missing_values)
    # 2- mapping the codes to the desired ones
Commodity_Policies.exporter = map(x -> get(exporter_mapping, x, x), Commodity_Policies.exporter);
Commodity_Policies.importer = map(x -> get(exporter_mapping, x, x), Commodity_Policies.importer);
    # 3- now there might be duplicated rows, so we need to remove them
unique!(Commodity_Policies);

# Adding Alberta to the Commodity_Policies table
canada_rows = Commodity_Policies[(coalesce.(Commodity_Policies.importer, -1) .== 124) .| (coalesce.(Commodity_Policies.exporter, -1) .== 124), :];
canada_rows.importer[coalesce.(canada_rows.importer, -1) .== 124] .= 9999;
canada_rows.exporter[coalesce.(canada_rows.exporter, -1) .== 124] .= 9999; #Using Canada data for Alberta
Commodity_Policies = vcat(Commodity_Policies, canada_rows);
Trade_data = leftjoin(Trade_data, Commodity_Policies, on = [:year,:importer, :hsCode, :exporter]);
Commodity_Policies = nothing

#**********
Distance = DataFrame(CSV.File("country_distance.csv"));
describe(Distance)
select!(Distance, Not([:iso_o, :iso_d]));
# mapping exporter codes to the ones in the Country_list
    # 1- ensuring that all the target values are in the dataset 
missing_values = setdiff(target_values, unique(Distance.Origin_PartnerCode))
println("Missing values: ", missing_values)
    # 2- mapping the codes to the desired ones
Distance.Origin_PartnerCode = map(x -> get(exporter_mapping, x, x), Distance.Origin_PartnerCode);
Distance.Destination_PartnerCode = map(x -> get(exporter_mapping, x, x), Distance.Destination_PartnerCode);
    # 3- now there might be duplicated rows, so we need to remove them
unique!(Distance);
# Adding Alberta to the Distance table
canada_rows = Distance[(coalesce.(Distance.Origin_PartnerCode, -1) .== 124) .| (coalesce.(Distance.Destination_PartnerCode, -1) .== 124), :];
canada_rows.Origin_PartnerCode[coalesce.(canada_rows.Origin_PartnerCode, -1) .== 124] .= 9999;
canada_rows.Destination_PartnerCode[coalesce.(canada_rows.Destination_PartnerCode, -1) .== 124] .= 9999; #Using Canada data for Alberta
Distance = vcat(Distance, canada_rows);
rename!(Distance, :Origin_PartnerCode => :importer, :Destination_PartnerCode => :exporter);
dropmissing!(Distance);
Trade_data = leftjoin(Trade_data, Distance, on = [:importer, :exporter]);
Distance = nothing
#**********
# Taking care of the importer side of the Macro_Var table
Macro_Var = DataFrame(CSV.File("Macro_Var.csv"));
# Let's include Alberta in the Macro_Var table
Macro_Var_Alberta = DataFrame(CSV.File("Macro_Var_Alberta.csv"));
Macro_Var = vcat(Macro_Var, Macro_Var_Alberta);
describe(Macro_Var)
select!(Macro_Var, Not([:Country_ISO_3]));
for name in names(Macro_Var)
    rename!(Macro_Var, name => "$(name)_importer")
end
rename!(Macro_Var, "Country Code_importer" => :importer, "year_importer" => :year);
# mapping importer codes to the ones in the Country_list
    # 1- ensuring that all the target values are in the dataset
missing_values = setdiff(target_values, unique(Macro_Var.importer));
println("Missing values: ", missing_values)
    # 2- mapping the codes to the desired ones
Macro_Var.importer = map(x -> get(exporter_mapping, x, x), Macro_Var.importer);
    # 3- now there might be duplicated rows, so we need to remove them
unique!(Macro_Var);
Trade_data = leftjoin(Trade_data, Macro_Var, on = [:year, :importer]);


# Taking care of the exporter side of the Macro_Var table
Macro_Var = DataFrame(CSV.File("Macro_Var.csv"));
Macro_Var_Alberta = DataFrame(CSV.File("Macro_Var_Alberta.csv"));
Macro_Var = vcat(Macro_Var, Macro_Var_Alberta);
select!(Macro_Var, Not([:Country_ISO_3]));
for name in names(Macro_Var)
    rename!(Macro_Var, name => "$(name)_exporter")
end
rename!(Macro_Var, "Country Code_exporter" => :exporter, "year_exporter" => :year);
# mapping importer codes to the ones in the Country_list
    # 1- ensuring that all the target values are in the dataset
missing_values = setdiff(target_values, unique(Macro_Var.exporter));
println("Missing values: ", missing_values)
    # 2- mapping the codes to the desired ones
Macro_Var.exporter = map(x -> get(exporter_mapping, x, x), Macro_Var.exporter);
    # 3- now there might be duplicated rows, so we need to remove them
unique!(Macro_Var);


Trade_data = leftjoin(Trade_data, Macro_Var, on = [:year, :exporter]);
Macro_Var = nothing
describe(Trade_data)
#**********
CSV.write("2- Diversification_Project_Raw.csv", Trade_data, writeheader=true)