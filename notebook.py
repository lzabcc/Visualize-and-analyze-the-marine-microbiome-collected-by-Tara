# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import altair as alt
#read the data
file_path_sample = "data\Tara_SampleMeta.csv"

file_path_OTU = "data\Tara_OTUtableTax_80CAb.csv"


imd_df_sample=pd.read_csv(file_path_sample)

trans_data = pd.read_csv('data\Tara_OTUtable_80CAb_transp.csv')

# # 计算每行第8列到最后一列的和

merged_data = pd.merge(imd_df_sample,trans_data,left_on='SampleID',right_on='Sample')

# 如果需要，可以删除多余的Sample列
merged_df = merged_data.drop(columns=['Sample'])
merged_data = merged_df.drop(columns=['unclassified'])



# #————————————————柱状图——————————————————————#
imd_df_OTU=pd.read_csv(file_path_OTU)
imd_df_OTU['sumAbundance'] = imd_df_OTU.iloc[:,7:].sum(axis=1)
imd_df_OTU['rank'] = imd_df_OTU['sumAbundance'].rank(ascending=False, method='dense')
import altair as alt

# Assuming you have loaded or defined the imd_df_OTU DataFrame
stacked_bar_chart = alt.Chart(imd_df_OTU).mark_bar().encode(
    x=alt.X('OTU_rep:N', sort='-y'),
    y=alt.Y('sumAbundance:Q', axis=alt.Axis(title='sum abudance'), scale=alt.Scale(zero=False)),
    tooltip=['OTU_rep:N', 'sumAbundance:Q']
).transform_calculate(
    keep_otu=alt.expr.if_(
        (alt.datum.OTU_rep != 'unclassified') & (alt.datum.OTU_rep != 'unclassified_other'),
        alt.datum.OTU_rep,
        None
    )
).transform_filter(
    alt.datum.keep_otu != None
).transform_window(
    rank='rank()'
).transform_filter(
    alt.datum.rank <= 50
)

# Create a single selection dropdown menu
category_dropdown = alt.selection_point(
    fields=['category'],
    name='Select',
    bind=alt.binding_select(options=['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus']),
)

# Apply the selection to the chart
filtered_chart = stacked_bar_chart.add_params(category_dropdown).transform_calculate(
    category=alt.expr.if_(
        category_dropdown['category'] != 'Class',
        alt.datum[category_dropdown['category']],
        'Genus'
    )
).transform_filter(
    alt.datum.category != 'Genus'
).encode(
    color=alt.Color('category:N', title='Selected Category', scale=alt.Scale(scheme='category20'))
).properties(
    width=600,
    height=200,
    title='Chart with different category'  # Initial chart title
)


# # Display the filtered chart with a manually specified legend title
#filtered_chart.save("barchartofsumAbundance-category.html")

imd_df_OTU['rank'] = imd_df_OTU['sumAbundance'].rank(ascending=False, method='dense')



# Create a single selection dropdown menu
range_dropdown = alt.selection_point(
    name='Select',
    fields=['range'],
    bind=alt.binding_select(options=[-10, 10, 20])
)

last10= int(imd_df_OTU['rank'].iloc[-10])
#print (type(last10))
rank_bar_chart = alt.Chart(imd_df_OTU).mark_bar().encode(
    x=alt.X('OTU_rep:N', sort='-y'),
    y=alt.Y('sumAbundance:Q', axis=alt.Axis(title='sum abudance'), scale=alt.Scale(zero=False)),
    tooltip=['OTU_rep:N', 'sumAbundance:Q']
).transform_filter(
 (alt.datum.OTU_rep != 'unclassified') 
).add_params(range_dropdown).transform_filter(
    alt.expr.if_(
        range_dropdown['range'] >= 10,
        alt.datum.rank <= alt.expr.toNumber(range_dropdown['range']),
        alt.datum.rank >= last10
    )
).encode(
    #color=alt.Color('Genus:N', title='Selected Category', scale=alt.Scale(scheme='category20'))
).properties(
    width=600,
    height=200,
    title='Chart with different rank'  # Initial chart title
)
barchart=(filtered_chart &  rank_bar_chart).save('barChartOfSumAbundance-Rank&Category.html')

#-------------------地图-------------------------#
#--------------长转宽--------------#
merged_data.rename(columns={
    'Latitude[degreesNorth]': 'Latitude',
    'Longitude[degreesEast]': 'Longitude',
    'SamplingDepth[m]':'SamplingDepth'
}, inplace=True)

merged_data['sumAbundance'] = merged_data.iloc[:,10:].sum(axis=1)
merged_data['sumAbundance']


columns_except_last = merged_data.columns[:-1].tolist()
# Get the last column name
last_column = merged_data.columns[-1]
# Reorder the columns
new_order = columns_except_last[:9] + [last_column] + columns_except_last[10:]
# Create a new DataFrame with the reordered columns
reordered_data = merged_data[new_order]
merged_data = reordered_data


columns_to_melt = merged_data.columns[10:].to_numpy()
# 使用 Pandas 的 melt 函数将数据转换为长格式
melted_data = pd.melt(
    merged_data,
    id_vars=list(merged_data.columns[:10]),  # 保持不变的列
    value_vars=columns_to_melt,  # 需要转换的列
    var_name='OTU_rep',  # 新的变量列的名称
    value_name='Value'  # 新的值列的名称
)
#melted_data.to_excel('melted_data1.xlsx')


#------------------bar chart for test------------------------#



# columns_to_aggregate = melted_data.columns[0:9]

# # 使用 groupby 和 agg 将前10列按顺序放到一个数组中
# ordered_data = melted_data.groupby('OTU_rep', as_index=False).agg({
#     'Value': 'sum',  # 对 'Value' 列求和
#     **{col: list for col in columns_to_aggregate}  # 对前10列使用 list 函数
# })

# ordered_data['rank'] = ordered_data['Value'].rank(ascending=False, method='first')
# #ordered_data.to_excel('oreder_data.xlsx')
# # 创建水平条形图
# bars_test = alt.Chart(ordered_data).mark_bar(orient='horizontal').encode(
#     y=alt.Y('OTU_rep:N', sort='-x'),  # OTU_rep 作为 y 轴，并按 x 轴（Value）降序排序
#     x='Value:Q',  # Value 作为 x 轴
#     tooltip=("Value")
# ).transform_filter(
#     alt.datum.rank <= 20  # 筛选前20个最大的值
# )
# # 保存图表为 HTML 文件
# bars_test.save('bars_test.html')


# #-------------------数据处理-----------------------#

# #merged_data.to_excel('merged_data.xlsx')

#-------------------开始绘图-----------------------#

import altair as alt
from vega_datasets import data
import geopandas as gpd

# 加载世界地图的 GeoJSON 数据
oceans = gpd.read_file(data.world_110m.url, driver="TopoJSON")


backgroundMap = alt.Chart(oceans).mark_geoshape(
    fill='lightgray',
    stroke='white'
).properties(
    width=800,
    height=500
).project('equirectangular')

brush = alt.selection_interval()

points = alt.Chart(melted_data).mark_point().encode(
    longitude='Longitude:Q',
    latitude='Latitude:Q',
    tooltip=['SampleID:N', 'OTU_rep:N','Value:Q'],
    #tooltip=['SampleID:N', 'sumAbundance:Q'],
    shape='OceanAndSeaRegion:N',
    color=alt.condition(brush, alt.Color('MarinePelagicBiomes:N'), alt.value('grey')),
    size=alt.Size('Value:Q'),
    #size=alt.Size('sumAbundance:Q'),
).properties(
    width=800,
    height=500
).add_params(brush)


# Display all map layers and points
left=alt.layer(
    backgroundMap,
    points
).properties(title='Map')
left.save("left.html")

#print(columns_to_fold)
# 使用Altair的transform_fold()方法将数据转换为长格式


bars_Province = alt.Chart(melted_data).transform_joinaggregate(
    cumulative_value='sum(Value)',
    groupby=['OTU_rep']
).mark_bar(orient='horizontal').encode(
    y='OTU_rep:N',
    x="Value:Q",
    color='MarinePelagicBiomes:N',
    tooltip=['SampleID:N','OTU_rep:N','rank:Q',"cumulative_value:Q"]
).transform_window(
    rank='rank(cumulative_value)',
    sort=[alt.SortField('cumulative_value', order='descending')],
).transform_filter(alt.datum.rank<=1351).properties(title='Total Abundance in some region')





# Assuming melted_data is your DataFrame
bars_depth = alt.Chart(melted_data).transform_joinaggregate(
    cumulative_value='sum(sumAbundance)',
    groupby=['OTU_rep']
).transform_calculate(
    cumulative_value_divided= ((alt.datum.cumulative_value) / 176040)
).mark_bar().encode(
    x=alt.X("SamplingDepth:Q", bin=alt.Bin(extent=[5, 1000], step=100)),
    y=alt.Y("cumulative_value_divided:Q", title="Cumulative Value "),
    color=alt.value("blue")
)


brush1=alt.selection_interval(encodings=['x'])

bars_overlay = bars_depth.encode(color=alt.value("red")).transform_filter(brush)
medium_bars = alt.layer(bars_depth, bars_overlay)
medium_bars1=alt.hconcat(
  medium_bars.encode(
    x=alt.X("SamplingDepth:Q", bin=alt.Bin(extent=[5, 1000], step=100)),
    tooltip=['SampleID:N', 'sumAbundance:Q']
  ).add_params(brush1),
  medium_bars.encode(
    x=alt.X("SamplingDepth:Q",).bin(maxbins=50, extent=brush).scale(domain=brush1),
    tooltip=['SampleID:N', 'sumAbundance:Q']
  ),
).properties( title='Sample size with depth')


# combine layers for histogram
right_bars = alt.layer(bars_Province,bars_Province.encode(color=alt.value("goldenrod")).transform_filter(brush))

# 使用刷选区域内的数据生成饼图
PieChart = alt.Chart(melted_data).mark_arc().encode(
    theta='Value:Q',
    color="MarinePelagicBiomes:N",
    tooltip=['OTU_rep:N', 'Value:Q']
).properties(width=300, height=300, title='Marine Pelagic Biomes of Selected range ').transform_filter(brush)


# 将图表组合在一起
#(left & Piechart).save("地图.html")

#Map = (left & (medium_bars1 | ( PieChart))).resolve_legend()

Map = (left & (medium_bars1 | ( PieChart & right_bars )))
#Map = (left & (medium_bars | medium_bars_100 | (right_bars & PieChart))).resolve_legend()
Map.save("map_mcv.html")


# #----------------------随机采样---------------------------#

import random
import altair as alt
import pandas as pd

# 假设 merged_data 是你的数据框架

# 选择列的范围（从第10列到倒数第二列）
#selected_columns = list(merged_data.columns)[9:-1]
selected_columns = random.sample(list(merged_data.columns)[9:-1], k=5)  # 选择两列，你可以根据需要修改 k 的值
#selected_columns.append('SamplingDepth')
# 使用选择的列进行 transform_fold
random_sampling = alt.Chart(merged_data).transform_window(
    index='count()'
).transform_fold(
    selected_columns
).mark_line().encode(
    x='key:N',
    y='value:Q',
    color='OceanAndSeaRegion:N',
    detail='index:N',
    opacity=alt.value(0.5)
).properties(width=500, title='Sampling Depth').resolve_legend()

random_sampling.save("random sampling.html")

#merged_data['date'] = pd.to_datetime(merged_data[['Year', 'Month']].assign(DAY=1))



#--------------------PCA--------------------------------#
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import altair as alt

# 假设 merged_data 包含了样本元数据和OTU数据

# 提取样本元数据和OTU数据
sample_meta = merged_data.iloc[:, :10]

otu_data = merged_data.iloc[:, 10:-1] 
# 对OTU数据进行标准化
otu_data_standardized = StandardScaler().fit_transform(otu_data)

# 对OTU数据进行PCA降维
pca = PCA(n_components=2)
pca_result = pca.fit_transform(otu_data_standardized)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# 合并样本元数据和降维后的主成分
final_data = pd.concat([sample_meta, pca_df], axis=1)


catSelection = alt.selection_point(fields=['LayerOfOrigin'],on='click')

#set up a conditional colouring based on selection
catColor = alt.condition(catSelection,
 alt.Color('LayerOfOrigin:N', legend=None), #hide the legend
 alt.value('lightgray')
)

# 使用Altair绘制散点图
chart_PCA = alt.Chart(final_data).mark_point().encode(
    x='PC1:Q',
    y='PC2:Q',
    color=alt.Color('LayerOfOrigin:N'),
    tooltip=['SampleID:N', 'Latitude:Q', 'Longitude:Q', 'MarinePelagicProvince:N']
).properties(
    title='PCA of OTU Data with LayerOfOrigin'
).transform_filter( #transform data by filtering using catSelection
 catSelection
)


# 保存降维后的数据到Excel文件
#final_data.to_excel('pca_result_with_metadata.xlsx', index=False)

# 显示图表
chart_PCA.save('PCA.html')


legend = alt.Chart(final_data).mark_point().encode(
 y=alt.Y('LayerOfOrigin',axis=alt.Axis(orient='right')),
 color=catColor,
).add_params(
 catSelection
)

PCA=(chart_PCA|legend).resolve_legend()
PCA.save('PCA.html')

# mvc = (
#  #filtered_chart|
#  Map &
#  random_sampling&
#  PCA
#  #resolve_legend='independent'
# )# title='Index of Multiple Deprivation Dashboard'
# mvc.save('MVC of Operational Taxonomic Units.html')



#-------------------merge data and map-----------------------#


# import altair as alt
# from vega_datasets import data
# import geopandas as gpd

# # 加载世界地图的 GeoJSON 数据
# oceans = gpd.read_file(data.world_110m.url, driver="TopoJSON")


# backgroundMap = alt.Chart(oceans).mark_geoshape(
#     fill='lightgray',
#     stroke='white'
# ).properties(
#     width=800,
#     height=500
# ).project('equirectangular')

# brush = alt.selection_interval()

# points = alt.Chart(merged_data).mark_point().encode(
#     longitude='Longitude:Q',
#     latitude='Latitude:Q',
#     tooltip=['SampleID:N', 'sumAbundance:Q'],

#     shape='OceanAndSeaRegion:N',
#     color=alt.condition(brush, alt.Color('MarinePelagicBiomes:N'), alt.value('grey')),
#     size=alt.Size('sumAbundance:Q'),

# ).properties(
#     width=800,
#     height=500
# ).add_params(brush)


# # Display all map layers and points
# left=alt.layer(
#     backgroundMap,
#     points
# )
# left.save("left.html")

# #print(columns_to_fold)

# bars_Province = alt.Chart(merged_data).mark_bar(orient='horizontal').encode(
#   y=alt.Y("MarinePelagicProvince:N", sort=alt.EncodingSortField(field="sumAbundance:Q", op="sum", order="descending")),
#   x="sumAbundance:Q",
#   color=alt.value("steelblue"),
#   tooltip=['SampleID:N',"sum_abundance:Q"]
# ).transform_filter(brush)

# #bars_depth = alt.Chart(merged_data).mark_bar().encode(
# bars_depth = alt.Chart(merged_data).mark_bar().encode(
#     x=alt.X("SamplingDepth:Q", bin=alt.Bin(extent=[5, 1000], step=100)),
#     y="sumAbundance:Q",
#     color=alt.value("steelblue")
# )

# brush1=alt.selection_interval(encodings=['x'])

# bars_overlay = bars_depth.encode(color=alt.value("goldenrod")).transform_filter(brush)
# medium_bars = alt.layer(bars_depth, bars_overlay)


# medium_bars1=alt.hconcat(
#   medium_bars.encode(
#     x=alt.X("SamplingDepth:Q", bin=alt.Bin(extent=[5, 1000], step=100)),
#     tooltip=['SampleID:N', 'sumAbundance:Q']
#   ).add_params(brush1),
#   medium_bars.encode(
#     x=alt.X("SamplingDepth:Q",).bin(maxbins=50, extent=brush).scale(domain=brush1),
#     tooltip=['SampleID:N', 'sumAbundance:Q']
#   ),
# )

# # combine layers for histogram
# right_bars = alt.layer(bars_Province)#, bars_overlay)

# # 使用刷选区域内的数据生成饼图
# PieChart = alt.Chart(merged_data).mark_arc().encode(
#     theta='sumAbundance:Q',
#     color="MarinePelagicBiomes:N",
#     tooltip=['SampleID:N', 'sumAbundance:Q']
# ).properties(width=300, height=300, title='Marine Pelagic Biomes of Selected range ').transform_filter(brush)


# # 将图表组合在一起
# #(left & Piechart).save("地图.html")

# #Map = (left & (medium_bars1 | ( PieChart))).resolve_legend()
# Map = (left & (medium_bars1 | ( PieChart & right_bars )))
# #Map = (left & (medium_bars1 | ( PieChart & right_bars )))

# Map.save("maps-detectionCapability.html")




