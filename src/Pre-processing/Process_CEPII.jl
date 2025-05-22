# Shift + Enter to run the code
using CSV
using DataFrames
import Statistics as stat 
cd(joinpath(@__DIR__, "data//CEPII"))

starting_year = 2013


#df = DataFrame(CSV.File("BACI_HS12_Y2012_V202501.csv"))
pattern = r"\_(?P<HSType>.{4})\_Y(?P<Year>\d{4})\_(?P<VintagePoint>.*)\_(?P<Alberta_Included>.*)\.(?P<FileFormat>.*)"
file_names = match.(pattern, readdir())
file_names = filter(!isnothing, file_names)
file_names = filter(x ->  parse(Int, x["Year"]) >= starting_year && x["Alberta_Included"] == "alberta" && x["HSType"] == "HS12" && x["FileFormat"] == "csv", file_names)
dfs = [DataFrame(CSV.File("BACI"*file_name.match)) for file_name in file_names]
df = vcat(dfs...)
df = unique(df)
rename!(df, Dict(:t => :year, :i => :exporter, :j => :importer, :k => :hsCode, :v => :value, :q => :quantity))
#Removing rows where value = 0 
df = df[df.value .!= 0, :]


# Change the quantity to missing values if equal to 0
# Alberta values have 0s while the rest of the data has missing values
df.quantity[coalesce.(df.quantity, 0.0) .== 0] .= missing;

# Going to HS4 Level:
df.hsCode = [x[1:4] for x in string.(df.hsCode, base=10, pad=6)];
df.UnitValueTimesvalue = df.value .* df.value./df.quantity
# Julia in contrast to Python, does not consider missing values 0 when groupby is used.
# Groupby year, hsCode, exporter, importer
df = combine(groupby(df, [:year, :hsCode, :exporter, :importer]), 
    :value => sum => :value, 
    :quantity => sum => :quantity, 
    :UnitValueTimesvalue => sum => :UnitValueTimesvalue)
df = transform(df, :UnitValueTimesvalue => (x -> x./df.value) => :AvgUnitPrice);
Imports = df[:, [:year, :hsCode, :exporter, :importer, :value, :AvgUnitPrice]];
ImportsfromWorld = Imports;
ImportsfromWorld.UnitValueTimesvalue = ImportsfromWorld[:, :value] .* ImportsfromWorld[:, :AvgUnitPrice]
ImportsfromWorld = combine(groupby(ImportsfromWorld, [:year, :hsCode, :importer]), 
    :value => sum => :value, 
    :UnitValueTimesvalue => sum => :UnitValueTimesvalue)
ImportsfromWorld = transform(ImportsfromWorld, :UnitValueTimesvalue => (x -> x./ImportsfromWorld.value) => :AvgUnitPriceofImporterFromWorld);
ExportstoWorld = Imports;
ExportstoWorld.UnitValueTimesvalue = ExportstoWorld[:, :value] .* ExportstoWorld[:, :AvgUnitPrice];
ExportstoWorld = combine(groupby(ExportstoWorld, [:year, :hsCode, :exporter]), 
    :value => sum => :value, 
    :UnitValueTimesvalue => sum => :UnitValueTimesvalue)
ExportstoWorld = transform(ExportstoWorld, [:UnitValueTimesvalue, :value] => (ByRow((x,y) -> x/y)) => :AvgUnitPriceofExporterToWorld);
rename!(ExportstoWorld, :value => :TotalExportofCmdbyExporter)
select!(ExportstoWorld, Not(:UnitValueTimesvalue))
rename!(ImportsfromWorld, :value => :TotalImportofCmdbyReporter)
select!(ImportsfromWorld, Not(:UnitValueTimesvalue))
# Merging Imports and ImportsfromWorld
Trade_data = leftjoin(Imports, ImportsfromWorld, on = [:year, :hsCode, :importer])
Trade_data = leftjoin(Trade_data, ExportstoWorld, on = [:year, :hsCode, :exporter])
Imports, ImportsfromWorld, df, ExportstoWorld = nothing, nothing, nothing, nothing

select!(Trade_data, Not(:UnitValueTimesvalue));
sort!(Trade_data, [:year, :importer, :exporter, :hsCode]);


# This function calculates the RCA (Revealed Comparative Advantage)
# https://unctadstat.unctad.org/EN/RcaRadar.html
function RCA(data)
    exporter_product = combine(groupby(data, [:year, :hsCode, :exporter]), 
    :value => sum => :TotalExportofCmdbyPartner)
    exporter = combine(groupby(data, [:year, :exporter]), 
    :value => sum => :TotalExportbyPartner)
    world_product = combine(groupby(data, [:year, :hsCode]), 
    :value => sum => :WorldExportofCmd)
    world = combine(groupby(data, [:year]), 
    :value => sum => :WorldTotalExport)
    data = leftjoin(data, exporter_product, on = [:year, :hsCode, :exporter])
    data = leftjoin(data, exporter, on = [:year, :exporter])
    data = leftjoin(data, world_product, on = [:year, :hsCode]) 
    data = leftjoin(data, world, on = [:year])
    data = transform(data, [:TotalExportofCmdbyPartner, :TotalExportbyPartner, :WorldExportofCmd, :WorldTotalExport] => ByRow((a, b, c, d) -> (a / b) / (c/ d )) => :Partner_Revealed_Comparative_Advantage)
    return data
end

# This function calculates the Theil Concentration Index for the exporter
# https://www.un.org/development/desa/dpad/wp-content/uploads/sites/45/CDP-bp-2023-59.pdf
function Theil_exporter_concentration(data)
    # It would have been better to define n, m for each year
    # However, it does not matter as n, m dont change from a year to another except for 1 unit! (from 2013 to 2023 investigated)
    n = length(unique(data.hsCode))
    m = length(unique(data.importer))
    # The index will range from 0 (least concentrated) to np.log(n*m) (least diversified)
    println("Minimum and Maximum Theil Exporter/Importer Concentration Index: [0, $(round(log(n*m), digits = 2))]")
    total_export_by_partner = combine(groupby(data, [:year, :exporter]), 
        :value => sum => :TotalExportbyPartner)
    total_export_by_partner_of_cmd = combine(groupby(data, [:year, :exporter, :hsCode]), 
        :value => sum => :TotalExportbyPartnerofCmd)
    total_export_by_partner_of_cmd = leftjoin(total_export_by_partner_of_cmd, total_export_by_partner, on = [:year, :exporter])
    total_export_by_partner_of_cmd = transform(total_export_by_partner_of_cmd, [:TotalExportbyPartnerofCmd, :TotalExportbyPartner] => 
                                                ByRow((a, b) -> a/b) => :x_k)
    total_export_by_partner_of_cmd = transform(total_export_by_partner_of_cmd, [:x_k] => ByRow((a) -> a * log(a)) => :x_k_times_ln_x)
    T_p = combine(groupby(total_export_by_partner_of_cmd, [:year, :exporter]), 
        :x_k_times_ln_x => sum => :x_k_times_ln_x)
    T_p[:, :T_p] = T_p[:, :x_k_times_ln_x] .+ log(n)#Product concentration
    select!(T_p, Not(:x_k_times_ln_x))
    # *******
    T_m = leftjoin(data, total_export_by_partner_of_cmd[:, [:year, :exporter, :hsCode, :TotalExportbyPartnerofCmd]], on = [:year, :exporter, :hsCode])
    T_m = transform(T_m, [:value, :TotalExportbyPartnerofCmd] => ByRow((a, b) -> a/b) => :x_j_k)
    T_m = transform(T_m, [:x_j_k] => ByRow((a) -> a * log(a * m)) => :x_j_k_times_ln_x)
    T_m = combine(groupby(T_m, [:year, :exporter, :hsCode]), 
        :x_j_k_times_ln_x => sum => :A)
    T_m = leftjoin(T_m, total_export_by_partner_of_cmd[:, [:year, :exporter, :hsCode, :x_k]], on = [:year, :exporter, :hsCode])
    T_m = transform(T_m, [:A, :x_k] => ByRow((a, b) -> a*b) => :B)
    T_m = combine(groupby(T_m, [:year, :exporter]), 
        :B => sum => :T_m)
    # *******
    data = leftjoin(data, T_p, on = [:year, :exporter])
    data = leftjoin(data, T_m, on = [:year, :exporter])
    data = transform(data, [:T_p, :T_m] => ByRow((a, b) -> a+b) => :Theil_Exporter_Concentration)
    return data
end

# This function calculates the Theil Concentration Index for the importer
# https://www.un.org/development/desa/dpad/wp-content/uploads/sites/45/CDP-bp-2023-59.pdf
function Theil_importer_concentration(data)
    n = length(unique(data.hsCode))
    m = length(unique(data.exporter))
    println(n, " ", m)
    # Import of Product k
    # From Market j
    Psi = combine(groupby(data, [:year, :importer]), 
        :value => sum => :psi)
    Psi = transform(Psi, [:psi] => ByRow((a) -> a/(n*m)) => :psi)
    println(Psi[Psi.importer .== 124, :])
    Psi = leftjoin(data, Psi, on = [:year, :importer])
    Psi = transform(Psi, [:value, :psi] => ByRow((a, b) -> (a/b) * (log(a/b))) => :x)
    Psi = combine(groupby(Psi, [:year, :importer]), 
        :x => sum => :x)
    Psi = transform(Psi, [:x] => ByRow((a) -> a/(n*m)) => :Theil_Importer_Concentration)
    data = leftjoin(data, Psi[:, [:year, :importer, :Theil_Importer_Concentration]], on = [:year, :importer])
    return data
end

# This function calculates the Product Complexity Index
# https://atlas.hks.harvard.edu/glossary
# https://www.pnas.org/doi/epdf/10.1073/pnas.0900943106
function Product_Complexity_Index(data)
    println("Product Complexity Index")
end

# This function calculates the Product Relatedness Index
# Prodcut relatedness is defined for a pair of products! Relatedness(Produckt_i, Product_j)!
# https://www.cepii.fr/PDF_PUB/wp/2012/wp2012-27.pdf
function Product_Relatedness_Index(data)
    println("Product Relatedness Index")
end

function Trade_Complementarity(data)
    Importer = combine(groupby(data, [:year, :importer]), 
        :value => sum => :TotalImportbyImporter)
    Exporter = combine(groupby(data, [:year, :exporter]), 
        :value => sum => :TotalExportbyExporter)
    data = leftjoin(data, Importer, on = [:year, :importer])
    data = leftjoin(data, Exporter, on = [:year, :exporter])
    data = transform(data, [:TotalImportofCmdbyReporter, :TotalImportbyImporter,:TotalExportofCmdbyPartner, :TotalExportbyExporter] 
            => ByRow((a, b, c, d) -> abs((a / b)-(c /d)) ) => :Trade_Complementarity)
    select!(data, Not([:TotalImportbyImporter, :TotalExportbyExporter]))
    return data
end

#Trade_data2 = RCA(Trade_data)
Trade_data2 = RCA(Trade_data);
Trade_data3 = Theil_exporter_concentration(Trade_data2);
Trade_data4 = Theil_importer_concentration(Trade_data3);
Trade_data5 = Trade_Complementarity(Trade_data4);
select!(Trade_data5, [:year, :importer, :exporter, :hsCode, :value, :AvgUnitPrice, 
                        :AvgUnitPriceofImporterFromWorld,:TotalImportofCmdbyReporter,:AvgUnitPriceofExporterToWorld, :TotalExportofCmdbyPartner,  :Partner_Revealed_Comparative_Advantage, 
                                    :Theil_Exporter_Concentration, :Theil_Importer_Concentration, :Trade_Complementarity]);


Trade_data5

                                
# Saving the data
CSV.write("..//1- CEPII_Processed_HS4_$(starting_year)_2023.csv", Trade_data5, writeheader = true)

