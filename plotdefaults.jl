rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 8
    font0 = Dict(
    "font.size" => 10,          # title
    "axes.labelsize" => 9, # axes labels
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    "lines.linewidth" => 0.7,
    "lines.markersize" =>2.5,
    )
merge!(rcParams, font0)

nns = ["PCA-Net", "DeepONet", "PARA-Net", "FNO"]
sizes = [L"w = 16\,\,/\,\,d_f = 2",L"w = 64 \,\,/\,\ d_f = 4",L"w = 128 \,\,/\,\ d_f = 8",L"w = 256 \,\,/\,\ d_f = 16"]
colors = ["#3A637B", "#C4A46B", "#FF6917", "#D44141" ] # colorblind friendly pallet https://davidmathlogic.com/colorblind/#%233A637B-%23C4A46B-%23FF6917-%23D44141
markers = ["o", "s", "^", "*"]
linestyle = ["dotted", "-.", "--", "-", ]

nn_linefigsize = (6.5, 1.625)

# for median/worst case error case plotting
nn_names = ["PCA", "DeepONet", "PARA", "FNO"]