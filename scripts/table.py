import os
import shutil
import sys

import matplotlib.pyplot as plt

from triple_scint.fixedparams import Target, Observatory
from triple_scint.data import Data
from triple_scint.model import ModelPhen
from triple_scint.fit import FitPhen
from triple_scint.uncertainty import SamplePhen, SamplePhys, SamplePres
from triple_scint.utils import tex_uncertainties

from triple_scint.visualize import visualize_model_zoom, visualize_model_folded


tbl1_wrap_upper = (
    r"$ "
    # r"\renewcommand\arraystretch{1.33} "
    # r"\begin{array}{ R @{\quad} R @{\quad} R @{\quad} R @{\quad} R @{\quad} R @{\quad} R } "
    r"\begin{array}{lccccccc} "
    r"\# "
    r"& \mathrm{priors}"
    r"& i_\mathrm{p}"
    r"& \Omega_\mathrm{p}"
    r"& d_\mathrm{p}"
    r"& s"
    r"& \xi"
    r"& v_\mathrm{lens,\parallel} \\[0.5em] "
)
tbl1_wrap_lower = r" \end{array} $"


tbl2_wrap_upper = r"""\begin{table*}
    \caption{Constraints on free parameters}
    {
    \renewcommand{\arraystretch}{1.5}
    \begin{tabular}{lcccccccc}
    \hline
    \hline
        &
        Informative priors &
        Solutions\textsuperscript{*} &
        $i_\mathrm{p}$ &
        $\Omega_\mathrm{p}$ &
        $d_\mathrm{p}$ &
        $s$ &
        $\xi$ &
        $v_\mathrm{lens,\parallel}$
    \\
        &
        &
        &
        (deg) &
        (deg) &
        (kpc) &
        &
        (deg) &
        (km/s)
    \\
    \hline
"""
tbl2_wrap_lower = r"""    \hline
    \end{tabular}
    }
\end{table*}
"""


def get_table1_row(ifit, priors, sample):
    txt_all = f"{ifit + 1}"
    priors_txt = [sample.pardict[prior].symbol for prior in priors]
    priors_txt = ", ".join(priors_txt)
    txt_all += f" & {priors_txt}"
    for par, samp in sample.samp_dict.items():
        result = tex_uncertainties(samp.distribution.value, sample.weights)
        txt_all += rf" & {result} "

    txt_all = txt_all + r"\\[0.5em]"
    return txt_all


def get_table2_row(ifit, priors, nsol, sample):
    txt_all = ""
    txt_all += f"        Fit {ifit + 1} &\n"
    priors_txt = [sample.pardict[prior].symbol for prior in priors]
    priors_txt = "$, $".join(priors_txt)
    txt_all += f"        ${priors_txt}$ &\n"
    txt_all += f"        {nsol} &\n"
    for par, samp in sample.samp_dict.items():
        result = tex_uncertainties(samp.distribution.value, sample.weights)
        line = f"        ${result}$"
        if par != "v_lens":
            line += " &"
        txt_all += f"{line}\n"
    txt_all += "    \\\\\n"
    return txt_all


def latex2image(
    latex_expression, image_name, image_size_in=(9, 3), fontsize=16, dpi=200
):
    """
    A simple function to generate an image from a LaTeX language string.

    Parameters
    ----------
    latex_expression : str
        Equation in LaTeX markup language.
    image_name : str or path-like
        Full path or filename including filetype.
        Accepeted filetypes include: png, pdf, ps, eps and svg.
    image_size_in : tuple of float, optional
        Image size. Tuple which elements, in inches, are: (width_in, vertical_in).
    fontsize : float or str, optional
        Font size, that can be expressed as float or
        {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}.

    Returns
    -------
    fig : object
        Matplotlib figure object from the class: matplotlib.figure.Figure.

    """

    plt.rcParams.update({"text.usetex": True})

    fig = plt.figure(figsize=image_size_in, dpi=dpi)
    text = fig.text(
        x=0.5,
        y=0.5,
        s=latex_expression,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fontsize,
    )

    plt.savefig(image_name)

    return fig


if __name__ == "__main__":
    print("Start")
    scriptfile = sys.argv[0]
    outdir = sys.argv[1]
    datafilename = sys.argv[2]

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    shutil.copy(scriptfile, outdir)

    target = Target(ref_all="Archibald")
    observatory = Observatory(telescope="GBT")
    print("Loading data")
    data = Data(datafilename=datafilename)

    print("Setting up phenomenological model")
    model_phen = ModelPhen(target=target, observatory=observatory)
    fit_phen = FitPhen(model_phen, data, priors={})

    print("Fitting phenomenological model to data")
    fit_phen.find_equad()
    res_phen = fit_phen.optimize()
    fit_phen.print_result(res_phen)

    print("Making model plot with zooms")
    fig_zoom = visualize_model_zoom(model_phen, data, res_phen)
    plt.savefig(f"{outdir}/model_zoom.pdf")

    print("Making folded model plot")
    fig_fold = visualize_model_folded(model_phen, data, res_phen)
    plt.savefig(f"{outdir}/model_fold.pdf")

    nsol = [2, 2, 2, 1, 1]
    priors = [
        ["d_p"],
        ["i_p", "d_p"],
        ["i_p"],
        ["i_p", "omega_p"],
        ["i_p", "omega_p", "d_p"],
    ]

    tbl1_body = ""
    tbl2_body = ""

    sample_phen = SamplePhen(fit_phen, nmc=int(1e6))

    for ifit in range(5):
        print(f"Fit {ifit + 1}")

        sample_phys = SamplePhys()

        if ifit == 0:
            sample_phys.from_phen_and_parallax(sample_phen)

        if ifit == 1:
            sample_phys.from_phen_and_i_p(sample_phen)
            sample_phys.weight_by_parallax()

        if ifit == 2:
            sample_phys.from_phen_and_i_p(sample_phen)

        if ifit == 3:
            sample_phys.from_phen_and_i_p(sample_phen)
            sample_phys.weight_by_omega_p()

        if ifit == 4:
            sample_phys.from_phen_and_i_p(sample_phen)
            sample_phys.weight_by_omega_p()
            sample_phys.weight_by_parallax()

        sample_pres = SamplePres()
        sample_pres.from_phys(sample_phys)

        tbl1_body += get_table1_row(ifit, priors[ifit], sample_pres)
        tbl2_body += get_table2_row(ifit, priors[ifit], nsol[ifit], sample_pres)

    tbl1 = tbl1_wrap_upper + tbl1_body + tbl1_wrap_lower
    tbl2 = tbl2_wrap_upper + tbl2_body + tbl2_wrap_lower

    latex2image(tbl1, f"{outdir}/constraints-table.pdf")

    file1 = open(f"{outdir}/constraints-table.txt", "w")
    file1.write(tbl2)
    file1.close()
