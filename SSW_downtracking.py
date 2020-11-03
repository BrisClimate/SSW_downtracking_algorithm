'''
This code uses a polar cap height anomaly time-height slice  60 days either side of SSW onset to produce a 
downtracking path from onset date in the stratosphere to surface impact date. The descent rates can be adjusted for
models with limited vertical levels (eg. CMIP6 has only 8 stored levels at daily resolution) and works on reanalysis
data with a greater number of vertical levels.
'''

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import numpy as np
import xarray as xr
import dask

# open and read file using xarray
filename = ('CP07/new/SA_SSW_S_2019-01-02.nc')
PCH = xr.open_dataset(filename)['z']
z = PCH.squeeze()
z = z.transpose()


# pressure to height conversion
def pressure_to_height(pressure,surface_p=1013.25,H=7000):
    height= -H * np.log (pressure/surface_p)
    return height

# set up a dataset with additional coordinates
z=xr.DataArray.to_dataset(z)
z=z.assign_coords(time=range(-60,61))
#z=z.assign_coords(plev=z.plev/100)       # convert Pa to hPa for ERA40

heights = pressure_to_height(z.plev.data)
z=z.assign_coords(height=heights)


# set some descent rates (m per day) for atmosphere based on CHarlton and ONeill 09

# for CMIP6
if len(z.plev) == 8:
    U_strat_desc= 341
    L_strat_desc= 352
    trop_desc=    485

# for ERA reanalysis
else:
    U_strat_desc= 216
    L_strat_desc= 166
    trop_desc=    225

# identify index of max PCH at each level in a window
z_width=list(range(50,71))
max_days=[]
peak_indices=[]
for lev in range(0,len(z.plev)):
    z_max=np.argmax(z.z[lev,z_width])+z_width[0]    #index max value for level in range
    z_max=z_max.item(0)                             # extract index value from DataArray
    
    limit=121
    if len(z.plev) == 8:
        strat_cutoff = 50
        trop_cutoff = 250
    else:
        strat_cutoff = 30
        trop_cutoff = 200
    if z.plev[lev].data < strat_cutoff:
        delta=z.height[lev]-z.height[lev+1]
        days_add=(delta/U_strat_desc) +1
        days_add=round(days_add.item(0))
        if z_max+days_add < limit:
            end=z_max+days_add
        else:
            end = limit
    elif strat_cutoff <= z.plev[lev].data < trop_cutoff:
        delta=z.height[lev]-z.height[lev+1]
        days_add=(delta/L_strat_desc) +1
        days_add=round(days_add.item(0))
        if z_max+days_add < limit:
            end=z_max+days_add
        else:
            end = limit
    elif trop_cutoff <= z.plev[lev].data < 700:
        delta=z.height[lev]-z.height[lev+1]
        days_add=(delta/trop_desc) +1
        days_add=round(days_add.item(0))
        if z_max+days_add < limit:
            end =z_max+days_add
        else:
            end = limit
    elif 700 <= z.plev[lev].data <1000:
        days_add=1
        if z_max+days_add < limit:
            end =z_max+days_add
        else:
            end = limit
    z_width=list(range(z_max,end))     # reset window for next level
    print(len(z_width))
    max_days.append(z_max)             # add level index to the results

print(max_days)

#adjust max_days to x axis
max_days=np.array(max_days)
max_days=max_days-60

#set size and colours of scatter points, calculate lag
sizes = [9]*len(max_days)
colors=['blue']*len(max_days)
sizes[0],sizes[len(max_days)-1],colors[0],colors[len(max_days)-1] =36,36,'red','red'
lag=np.subtract(max_days[-1],max_days[0])

# set range and number of levels for PCH anomaly
clevs = np.linspace(-3.5, 3.5, 15)

#choose colormap, sensible levels and define a normalization
# instance which takes data values and translates them to levels
cmap = plt.get_cmap('RdBu_r')
norm = BoundaryNorm(clevs, ncolors=cmap.N, clip=True)
ax = plt.axes(ylim=(1,1000),xlabel=('days'),ylabel=('pressure level /hPa'))
fill = z.plot.contourf(ax=ax, levels=clevs,cmap=cmap,add_colorbar=False, add_labels=False)
cb = plt.colorbar(fill, orientation = 'horizontal', shrink=0.7, pad=0.3)
cb.set_label('PCH standardised anomaly', fontsize=12)

# add scatter plot to show downtracking
plt.scatter(max_days,z.plev,c=colors,s=sizes,zorder=1)
plt.plot(max_days,z.plev,zorder=2)
ax.set_title('PCH anomaly,SSW 06-01-2013, lag={}'.format(lag), fontsize=14)
ax.set_xticklabels(['-60','-40','-20','0','20','40','60'],rotation=0, horizontalalignment="center")

plt.yscale('log')
plt.gca().invert_yaxis()

plt.show()

