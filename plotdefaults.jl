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
markers = ["o", "s", "^", "*"]
linestyle = ["dotted", "-.", "--", "-", ]

nn_linefigsize = (6.5, 1.625)

# for median/worst case error case plotting
nn_names = ["PCA", "DeepONet", "PARA", "FNO"]
probs = ["Navier-Stokes", "Helmholtz", "Structural mechanics", "Advection"]

if coloroption == "paper"
    font0 = Dict(
    "figure.facecolor" => "#FFFFFF",
    "axes.facecolor" => "#FFFFFF",
    "savefig.facecolor" =>"#FFFFFF",
    )
    merge!(rcParams, font0)
    colors = ["#3A637B", "#C4A46B", "#FF6917", "#D44141" ] # colorblind friendly pallet https://davidmathlogic.com/colorblind/#%233A637B-%23C4A46B-%23FF6917-%23D44141
    lbl = "#000000"
    tk = "#808080"
elseif coloroption == "darkslides"
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    font0 = Dict(
    "figure.facecolor" => "#353F4F",
    "axes.facecolor" => "#353F4F",
    "savefig.facecolor" =>"#353F4F",
    )
    merge!(rcParams, font0)
    colors = ["#54ccff", "#C4A46B",  "#FF6917", "#D44141" ]
    lbl = "#E7E6E6"
    tk = "#E7E6E6"
end