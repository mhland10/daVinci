<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


# Overview of Topic

Rather than discussing the already established ideas in this area, I will simply reference you to [Geeks of Geeks](https://www.geeksforgeeks.org/comprehensive-guide-to-edge-detection-algorithms/), who do a good job at explaining the topic. One may notice that each of these algorithms are simply the convolution of a kernel that reflects the edge shape amongst the data. Thus, one can use the simple formulation:

$$
\epsilon_{edge}=f(t,\mathbf{x})*G
$$

Where $G$ is the edge kernel.

## Implementation

### The Discrete Wavelet Tranform

The Discrete Wavelet Transform (DWT) is more efficient than a convolution, $O(n)$ vs $O(n log(n))$. Thus, for large data sets the DWT method will be more desireable. Additionally, for instances where there may be noise, like experimental data, the DWT allows us to filter bands that contain the noise [1].


[1] Barbhuiya, A. H. M. J. I. (2018). *An Efficient Edge Detection Approach Using DWT.* International Journal of Computer Engineering & Technology, Vol 9, Iss 5, pgs 32-42. Available [here](http://www.iaeme.com/IJCET/issues.asp?JType=IJCET&VType=9&IType=5).


