import matplotlib.pyplot as plt
from scipy.stats import linregress



def plot_correlation(
    xdata, ydata, 
    xerr=None, 
    yerr=None, 
    xlabel=r'$\Delta\Delta G_\mathrm{expt}$ (kcal/mol)', 
    ylabel=r'$\Delta\Delta G_\mathrm{FEP}$ (kcal/mol)',
    ax=None,
    savefig=None
):

    xdata, ydata = np.array(xdata), np.array(ydata)
    
    stats = {
        'mue': np.mean(np.abs(xdata - ydata)),
        'mse': np.mean(ydata - xdata),
        'rmse': np.sqrt(np.mean((xdata - ydata)) ** 2)
    }
    slope, intercept, r_value, p_value, std_err = linregress(xdata, ydata)
    stats.update({
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'r2': r_value ** 2
    })

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 5))

    mae = np.mean(np.abs(xdata - ydata))
    mae_label = f'MAE: {mae:.2f}'

    r2 = r_value ** 2
    # r2_label = '$R^2: $' + f'{r2:.2f}'
    r2_label = '$R=$' + f'{r_value:.2f}'

    ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', label=mae_label, capsize=5)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    vmin, vmax = min([xmin, ymin]), max([xmax, ymax])
    ax.plot([vmin, vmax], [vmin, vmax], linestyle='--', color='black', label=r2_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xticks, yticks = ax.get_xticks(), ax.get_yticks()
    ticks = xticks if len(xticks) > len(yticks) else yticks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    ax.fill_between([vmin, vmax], [vmin-1, vmax-1], [vmin+1, vmax+1], color='gray', alpha=0.5)
    ax.fill_between([vmin, vmax], [vmin-2, vmax-2], [vmin+2, vmax+2], color='gray', alpha=0.3)
    ax.legend()

    return ax, stats
