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
Country_list = nothing
println("Number of Transactions among Alberta's top 30 trading partner: $(nrow(Trade_data))")
#**********
Commodity_Policies = DataFrame(CSV.File("cmd_policy.csv"));
describe(Commodity_Policies)
select!(Commodity_Policies, Not(:Name));
rename!(Commodity_Policies, :HS4 => :hsCode);
Trade_data = leftjoin(Trade_data, Commodity_Policies, on = [:year, :hsCode]);
Commodity_Policies = nothing
#**********
Distance = DataFrame(CSV.File("country_distance.csv"));
describe(Distance)
select!(Distance, Not([:iso_o, :iso_d]));
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
Trade_data = leftjoin(Trade_data, Macro_Var, on = [:year, :importer]);

Macro_Var = DataFrame(CSV.File("Macro_Var.csv"));
Macro_Var_Alberta = DataFrame(CSV.File("Macro_Var_Alberta.csv"));
Macro_Var = vcat(Macro_Var, Macro_Var_Alberta);
select!(Macro_Var, Not([:Country_ISO_3]));
for name in names(Macro_Var)
    rename!(Macro_Var, name => "$(name)_exporter")
end
rename!(Macro_Var, "Country Code_exporter" => :exporter, "year_exporter" => :year);
Trade_data = leftjoin(Trade_data, Macro_Var, on = [:year, :exporter]);
Macro_Var = nothing
describe(Trade_data)
#**********
CSV.write("2- Diversification_Project_Raw.csv", Trade_data, writeheader=true)