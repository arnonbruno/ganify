# GANify (v. 1.0.10)
<p align="center">
<img width="200" height="200" src="https://github.com/arnonbruno/ganify/blob/master/logo.png">
</p>

<b> Description: </b> GANify is an algorithm based on Generative Adversarial Learning to generate synthetic non-tensor data. The name GANify is an adaptation of acronym <b>GAN</b> (generative adversarial network) and Ampl<b>IFY</b>, meaning you can amplify the amount of data available with GANs .


<b> Installation: </b>
One can easily install GANify using the PIP:

<i>pip install ganify==1.0.10</i>


<b>How to use:</b>
Once installed, simply import the library and instatiate the model as described below:
<p align="center">
<img width="600" height="500" src="https://github.com/arnonbruno/ganify/blob/master/ganify.gif">
</p>

<b> Other info: </b>
The package also enables the creation of synthetic data using both <b> GANs </b> (Bengio et al., 2014) and <b> WGANs </b> (Chintala et al. 2017), by simply changing the argument <i>"type"</i> on <i>"fit_data"</i>
Additionally, you can view the model overall loss performance by calling the <i>"plot_performance()"</i> method after fit.

<b> Further improvements: </b>
Early stopping to optimize training interruption

<b> References: </b>
Generative Adversarial Nets (Bengio et al. 2014) - https://arxiv.org/pdf/1406.2661.pdf

Wasserstein GAN (Chintala et al. 2017) - https://arxiv.org/abs/1701.07875

Stabilizing Training of Generative Adversarial Networks through Regularization (Hofmann, 2017) - https://papers.nips.cc/paper/6797-stabilizing-training-of-generative-adversarial-networks-through-regularization.pdf

Improved Techniques for Training GANs (Chen et al., 2017) - https://arxiv.org/abs/1606.03498
