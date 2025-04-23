using DataFrames
using CSV
import Statistics as stat
cd("C:\\Users\\saeed.shadkam\\OneDrive - Government of Alberta\\Desktop\\ComTrade\\Other Tables\\")
starting_year = 2013

Trade_data = DataFrame(CSV.File("1- CEPII_Processed_HS4_$(starting_year)_2023.csv"));
#**********
Country_list = DataFrame(CSV.File("country_list.csv"));
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
rename!(Distance, :Origin_PartnerCode => :importer, :Destination_PartnerCode => :exporter);
dropmissing!(Distance);
Trade_data = leftjoin(Trade_data, Distance, on = [:importer, :exporter]);
Distance = nothing
#**********
Macro_Var = DataFrame(CSV.File("Macro_Var.csv"));
describe(Macro_Var)
select!(Macro_Var, Not([:Country_ISO_3]));
for name in names(Macro_Var)
    rename!(Macro_Var, name => "$(name)_importer")
end
rename!(Macro_Var, "Country Code_importer" => :importer, "year_importer" => :year);
Trade_data = leftjoin(Trade_data, Macro_Var, on = [:year, :importer]);

Macro_Var = DataFrame(CSV.File("Macro_Var.csv"));
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