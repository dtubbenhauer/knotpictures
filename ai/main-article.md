# On knot detection via picture recognition

Anne Dranowski, Yura Kabkov and Daniel Tubbenhauer

We would like to take a photograph of a knot and have a phone identify it by reading a diagram and computing its invariants. We start by counting crossings, give a plane-graph formula for the count, and show why four measurements of the ink recover nearly every label on familiar drawings but miss most labels on new ones.

## Introduction

Can we photograph a knotted cord and ask a phone which knot it is? We would like the phone to mark its reading on the photograph, so that we could follow the strand and inspect each overpass.

### From a photograph to a knot

We once needed precisely this check. Before the first version of this paper went on arXiv, we mistook a photograph of a trefoil for the figure-eight knot. The strand crossed itself four times in that view. In the usual tables the trefoil has three crossings and the figure-eight has four. Put a little curl in the usual trefoil diagram and it too has four. Now try the opposite experiment in Figure 1. Change which strand goes over at just one crossing. The crossing count stays at three, but the knot can now be untied.

[Vector illustration](../figures/switch-arcs.pdf)

A crossing change on the standard trefoil. Trace the ink between successive underpasses. The switch preserves the count, but permits two neighboring crossings to cancel and the remaining curl to straighten.

A *knot diagram* is a plane drawing of a closed strand with a choice of overpass at each crossing. With the diagram in hand one can compute knot invariants and compare them with a table. A braid word, for example, already records the overpasses. Gukov et al. trained a classifier to predict whether the closure of such a word is the unknot \[GHRS21\]. Starting from pixels adds the problem of reading the strands. Miller’s KnotFolio converts raster drawings to planar-diagram codes and computes a bracket polynomial \[Mil19\]. Liu and Zhang reconstructed a raster trefoil for their untangling interface \[LZ21\]. The accompanying appendix follows this passage from a picture to a diagram and then to the knot itself.

### What the experiments measure

Here we start with a question one can score directly on a drawing. Let $d(D)$ be the number of crossings in a diagram $D$. The *crossing number* $c(K)$ of its knot $K$ is the least value of $d(D)$ over all diagrams of $K$. Adding a curl is a Reidemeister-I move. It adds one to $d(D)$ and leaves $c(K)$ alone. A table of minimal diagrams assigns the same integer to both questions. A third way to get the integer right is to recognize a drawing one has seen before and recall its label.

Our first experiment reported $99.94\%$ accuracy for a convolutional neural network (CNN) on $1{,}811$ images held out from a collection of $9{,}051$ \[DKT25\]. Those images came from just $88$ original drawings. For this revision we built a classifier that takes only four inputs. It measures the amount of ink, how far it spreads from the center, how elongated it is, and how much lies far out. On labels represented by several drawings, it gets $99.92\%$ right on a fresh split by image, but only $21.66\%$ when we withhold each original together with all its descendants. A small CNN also loses accuracy on unseen drawings when we match the labels and resize settings in the two tests (Section 3C).

What would it take to count the crossings themselves? In Section 2A we derive a plane-graph formula that reads the count from the free ends, connected pieces of ink, and bounded regions. In Section 4 we put a little curl in each of a family of drawings. The knot stays the same, and the answer must go up by one. We test this prediction when gaps close or a stroke breaks.

## Counting crossings without finding them

### Read the ink

At a crossing in a textbook diagram the upper strand continues unbroken. A little white gap interrupts the lower strand. Imagine tracing the knot with scissors in hand and cutting out the short interval hidden at each underpass. If there are $n>0$ crossings, we have cut a circle in $n$ places, leaving $n$ arcs. Provided the marks are separated everywhere else, the number of connected pieces of ink is exactly $n$. One can recognize the crossing-free circle by eye because its single piece is closed.

What if a gap fills in? The two strands now meet at a solid four-way junction. One can keep counting even in a picture that mixes the two conventions. The count uses the ends of the strokes and the bounded regions of the paper. Figure 2 gives an example.

#### Proposition 2A.1.

Draw a generic knot projection with $n$ transverse crossings. At each crossing either cut a small gap in the lower strand or draw a solid four-way junction. Suppose no other strokes touch or break. Let $e$ be the number of free ends, $b$ the number of connected pieces of ink, and $f$ the number of bounded complementary regions. Then $$n=e-b+f.$$

#### Proof.

Regard the stroke centerlines as a plane graph. If $g$ crossings have gaps and $u$ are solid, then $e=2g$ and $n=g+u$. There are $e$ vertices of degree one and $u$ vertices of degree four. We may add vertices of degree two, for instance on a closed curve without a vertex. Summing the degrees gives $V-E=e/2-u$. Euler’s formula for a plane graph with $b$ components gives $V-E=b-f$. Hence $e-b+f=e/2+u=n$.

The same proof works for links with any number of closed strands. In particular a crossing-free circle gives $0-1+1=0$.

[Vector illustration](../figures/mixed.pdf)

Fill the three underpass gaps one at a time. The number of ends drops and faces appear. The combination in Proposition 2A.1 stays equal to three. Solid junctions forget which strand goes over.

In a fully gapped one-component diagram with $n>0$, every piece is an arc and $f=0$. At the other extreme, a connected *shadow* has no gaps at all. Then $e=0$ and $b=1$, so it has $n+1$ bounded regions. Both pictures tell us the crossing count. Only the gapped one tells us which strand passes over. For example, color each unbroken arc in the trefoil by an element of $\mathbb Z/3\mathbb Z$. At a crossing require the two underarc colors to sum to twice the overarc color. Each of the three equations becomes $x+y+z=0$, giving nine colorings. Switching the marked crossing yields an unknot, which has just the three constant colorings because boundary colorings extend uniquely across each Reidemeister move \[Prz98\]. To distinguish the two pictures the *incidence* of the arcs matters.

### A counter with no training

We tested the cut-circle rule on all fifteen entries in Jkasd’s illustrated table \[Jka08\], including the circle and one drawing of every prime knot through seven crossings. We cropped out the printed names, converted each crop to gray, and used Otsu’s threshold \[Ots79\] to separate ink from paper. Ink pixels can connect along an edge or diagonal. Background pixels connect along an edge, and padding joins the exterior into one background component. We return the ink-component count, except that one component enclosing exactly one hole is read as a circle. In a damaged raster a false loop can fool this last rule, just as a false gap can split an arc.

[Vector illustration](../figures/resolution-7-4.pdf)

The same <span class="math inline">7<sub>4</sub></span> diagram at three resolutions. The labels give the number of connected pieces of the thresholded ink. The smaller rasters are enlarged for display. Diagram adapted from Jkasd [Jka08].

At native size ($230\times230$ pixels per crop) the counter read all fifteen drawings correctly. It still read them all at side lengths $160$, $128$ and $96$. At $64$, $48$ and $32$ pixels it got respectively $11$, $5$ and $0$ of the fourteen nontrivial entries. It also read the circle correctly at every size. We used Lanczos downsampling and reapplied the threshold at each size. Shifting each native crop by zero or one pixel along each axis gave ranges of $11$–$12$ and $4$–$5$ correct at the two intermediate sizes. These are repeated measurements on the same fourteen drawings. In Figure 3 one can see the loss without knowing the name of the knot.

## What counts as a new knot picture?

### Where the 9,051 images came from

Philip Rideout drew a table of knots by starting from Gauss codes, laying out the corresponding plane graphs by circle packing, then painting a white strip under each overpass \[Rid20\]. The released training directory \[DKMT25\] contains views of $88$ original diagrams from this collection. The label is the crossing number of the table entry. The source counts are $$\begin{array}{c|rrrrrrrrr}
 \text{label} &0&3&4&5&6&7&8&9&10\\
 \text{source drawings} &1&1&1&2&3&7&21&49&3
\end{array}$$ For a label with $s$ drawings the generator made $\operatorname{round}(49/s)$ variants per drawing by choices of rotation and reflection (ties round to even, giving $24$ variants when $s=2$). Each of the resulting $431$ images was retained, resized twenty ways and padded back to $512\times512$ pixels, hence $431\cdot21=9{,}051$ files. Each of the $1{,}029$ three-crossing files has the same parent trefoil. The historical notebook \[Dra+25\] split the resulting list at random into $7{,}240$ training images and $1{,}811$ test images.

[Vector illustration](../figures/archive-views.pdf)

Three descendants of one six-crossing drawing and a second drawing with the same label. Strokes have been thickened uniformly for display. The released corpus [DKMT25] uses source geometry by Philip Rideout [Rid20].

We reconstructed the source groups from the order in which the notebook wrote the files. We retain anonymous source indices within each label, since directory iteration does not supply a reliable knot name. All $8{,}620$ resized descendants agreed with reconstructions from their assigned parents to within one grayscale level at every pixel. No PNG hash occurs in different reconstructed source groups. As a second check, all $199$ eligible transformed base images had their closest same-label radial ink profile in the assigned source group. There are $6{,}657$ distinct PNG files by SHA-256 among the $9{,}051$. The release also has a directory of $49$ test images, including $18$ exact byte copies from the larger directory. That directory is a separate collection from the historical $1{,}811$-image test. In the comparisons below we specify which images enter each score.

What does the cut-circle rule see in this material? We rasterized the same $88$ named vector drawings directly from the poster. At side lengths $64$, $128$, $256$, $512$, $1{,}024$ and $2{,}048$ pixels, it got $1$, $5$, $35$, $62$, $82$ and $88$ pictures right, respectively. In the archived training rasters it got just $65$ of $9{,}051$ right. The larger set has been skeletonized, transformed and class-balanced, so its image-weighted accuracy answers a different question from accuracy on the $88$ sources. Follow a single nine-crossing diagram through the preparation in Figure 5. Its ink breaks into $161$ pieces while its label stays put.

[Vector illustration](../figures/legacy_components.pdf)

One nine-crossing drawing before and after the archived image transformations. The colors mark connected pieces of foreground ink. Source geometry by Rideout [Rid20].

### Four measurements of the ink

Think of grayscale darkness as mass spread across the unit square. We measure its total mass $m$ (the mean darkness of the pixels), then normalize to a probability distribution. Let $X$ be the position of an ink sample relative to its center of mass. Put $S=\mathbb{E}(XX^{\mathsf T})$ and $t=\operatorname{tr}S$. Our four numbers are $$m, \qquad \sqrt{t}, \qquad
 \frac{(S_{11}-S_{22})^2+4S_{12}^2}{t^2}, \qquad
 \frac{\mathbb{E}\lVert X\rVert^4}{t^2}.$$ The third compares the covariance eigenvalues by taking their squared difference divided by the square of their sum. The fourth gives extra weight to ink far from the center. These are image moments of the kind used in pattern recognition since Hu \[Hu62\]. Centering removes translation, and rotation and reflection preserve all four in the continuum when no ink is clipped. Resizing the drawing within its frame changes the first two. Transformed relatives can consequently trace a recognizable family in this four-dimensional space.

Here is a geometric reason to doubt that these measurements count crossings. Replace a short arc inside a disk of radius $r$ by a thin Reidemeister-I curl, keeping the ink identical outside the disk. The crossing count changes by one. With darkness between zero and one, each unnormalized monomial moment through degree four changes by at most $\pi r^2$ on the unit square. Keep a fixed amount of well-separated ink elsewhere, so that $m$ and $t$ stay bounded away from zero. The four measurements, which are continuous functions of these moments, can be made arbitrarily close by shrinking the curl.

We standardize the four entries by their means and standard deviations in training and give each test image the label of its nearest training neighbor in Euclidean distance. We keep this rule throughout Table 1.

| Test population                            | Correct | Tested |  Exact |
|:-------------------------------------------|--------:|-------:|-------:|
| Random images, all nine labels             |   1,810 |  1,811 | 99.94% |
| Random images, labels 5 through 10         |   1,192 |  1,193 | 99.92% |
| Unseen source drawing, labels 5 through 10 |   1,292 |  5,964 | 21.66% |
| Independent illustrated table              |       0 |     15 |     0% |

Four ink measurements and one nearest neighbor. The second row restricts the first test to labels with several source drawings. For the last row we fit on the full training directory and test on the fifteen pictures from Section 2A.

The first split holds out a class-stratified $20\%$ of files (seed $42$). Of its $1{,}811$ test files, $621$ have an exact byte copy in training. Repeating the image split with seeds $0$ through $9$ gave perfect scores each time. Even keeping identical files together gave $99.89$–$100\%$ over ten splits by hash. Two resized copies of one source can have different hashes.

For the third row, withhold *every* descendant of one source, fit again, then test on that source. We can do this for each of the $85$ drawings with labels $5$ through $10$ while retaining the test label in training. The classifier missed every held-out image at labels five, six and seven, and got $513$ of the $1{,}029$ nine-crossing images right. Giving each tested source equal weight yields $36.42\%$ accuracy. Pooling all descendants gives the $21.66\%$ in the table. The $49$ nine-crossing drawings get much more weight in the first average. Since each source represents a different knot-table entry, holding out the source also holds out that knot type. On Jkasd’s independent fifteen-picture table the classifier called every entry a nine, including the circle. There the median fractional ink mass was $7.01\%$, compared with $0.43\%$ in the training directory.

This gives an explicit shortcut in the sense of Geirhos et al. \[GJM+20\]. A good score on the original task can be obtained from the gross distribution of ink alone. How the historical CNN made its decisions is a separate question. We could recover the training notebook, but the saved weights of that run were unavailable.

### Give a neural network the same test

We trained a new $2{,}893$-parameter CNN small enough to run on a CPU. We divided the $88$ sources into $57$ for fitting, $13$ for validation, and $18$ for testing. Labels $0$, $3$ and $4$ have a single source each, all used in fitting. From the fitting sources we used four of the $21$ scale settings per transformed variant, giving $1{,}192$ training images. We then assembled two sets of $1{,}054$ test images. One has unused sizes of the familiar fitting drawings. The other has views of the unseen drawings. The two sets have exactly the same joint distribution of crossing label and resize index. All $2{,}108$ test files have distinct hashes, and none occur among the fitting or validation files. Both tests cover labels $5$ through $10$.

The network takes an $80\times80$ grayscale image, applies three stride-two convolutions with ReLU activations, averages over four spatial quadrants, and predicts one number. We fit standardized labels by squared error, trained for $45$ epochs with Adam, and selected each run’s epoch by validation error on $52$ images from the validation sources. The three initializations share all drawings and test pictures. We rounded predictions to the nearest integer. We also refit the four-number classifier on precisely the same $1{,}192$ files.

|                       | Correct out of 1,054 |        | Mean absolute error |        |
|:----------------------|---------------------:|-------:|:-------------------:|:------:|
| 2-3(l)4-5             |             Familiar | Unseen |      Familiar       | Unseen |
| Four ink measurements |                1,011 |    169 |        0.04         |  1.94  |
| CNN, seed 17          |                  552 |    198 |        0.63         |  1.42  |
| CNN, seed 29          |                  341 |    118 |        0.94         |  1.79  |
| CNN, seed 43          |                  396 |    166 |        0.77         |  1.36  |

Holding label and resize frequencies fixed while changing which drawings are familiar. Errors are measured in crossings after rounding. All methods see the same training files and are scored on the same two test sets.

In all three runs the CNN lost accuracy on unseen drawings. Averaged across runs, it read $40.8\%$ of the familiar views correctly and $15.2\%$ of the unseen views. For the four ink measurements those rates were $95.9\%$ and $16.0\%$. On unfamiliar drawings the CNN’s mistakes were smaller on average than the nearest neighbor’s. The historical CNN and this small model differ in architecture, training set size, and input resolution. Table 2 measures the change for each fitted method on one source partition. Its $1{,}054$ images per arm are correlated views of a much smaller number of drawings.

## Put a curl in the picture

### Paired drawings

So far each source was a table entry. To vary the diagram while keeping the knot fixed, we generated pictures in pairs. Start with a closed two-strand braid with $k$ positive crossings, $k=1,3,5,7,9$. In one member of the pair insert a Reidemeister-I curl in an arc away from the crossings. The two pictures have $k$ and $k+1$ visible crossings. Before drawing any gaps, we enumerated transverse intersections of nonadjacent polygon segments. There were $k$ on the unaffected braid, zero between it and the replacement, and the prescribed zero or one on the replacement itself.

From $54$ geometric layout settings we reserved sixteen for evaluation, giving $80$ pairs. Each setting recurs for all five values of $k$ and both curl choices. Our unit of replication is a layout of these braid families. In the clean condition each picture is an antialiased gapped drawing at $160\times160$. In the second condition we add a five-pixel break away from any crossing. In the third we use thinner strokes and narrower crossing gaps, downsample to $72\times72$, then enlarge back to $160\times160$. The extra break can create a spurious arc, while the thin rendering can close a gap.

[Vector illustration](../figures/paired-comparison.pdf)

A five-crossing diagram and the same knot with a curl. An extra break splits an ink arc. Thin rendering can close the gaps. Colors in the bottom panels mark the connected components.

We also applied the archive-trained CNNs here with their preparation and weights fixed. All three missed all $128$ clean test images whose counts ($3$ through $10$) had appeared in training, and none gave the exact increase on any of their $64$ curl pairs. The regressors often returned numbers outside the training range. This test changes renderer and source geometry together, so it measures transfer across both changes.

On every thin image the ink-component counter returned one. Looking at the pictures suggested counting bounded white regions as one would in a shadow. This recovered the count in $125$ of $160$ images. We then tested the full formula $e-b+f$ retrospectively, including a fourth rendering in which an independent fair coin at each crossing chooses a gap or a solid junction (fixed random seeds). Of the $160$ pictures, $122$ actually mix both conventions. To estimate $e$, we thin the binary foreground by the Zhang–Suen algorithm \[ZS84\] and count skeleton pixels with exactly one of their eight neighbors present. The component and region counts use the original thresholded raster. No fitting is involved.

|                     | Correct absolute count out of 160 |             |         | Correct $+1$ out of 80 |
|:--------------------|----------------------------------:|------------:|--------:|-----------------------:|
| 2-4(l)5-5 Rendering |                    Ink pieces $b$ | Faces $f-1$ | $e-b+f$ |                $e-b+f$ |
| Clean gaps          |                               150 |           5 |     158 |                     80 |
| One extra break     |                                13 |           0 |       0 |                     80 |
| Thin strokes        |                                16 |         125 |     125 |                     77 |
| Random gaps         |                                23 |          17 |     146 |                     80 |

Three readings of the same $160$ diagrams in each condition. All underlying diagrams have at least one crossing. The last column tests whether adding the curl increases the prediction by exactly one. The face and full-formula comparisons are exploratory.

With one extra break, the full formula was wrong on all $160$ pictures yet detected the added curl in all $80$ pairs (Table 3). The ideal graph calculation predicts why. Cutting an ordinary stroke creates two ends and raises $b-f$ by one, so $e-b+f$ rises by one even though no crossing was added. In fact the estimate was one too high in $158$ pictures and two too high in the other two. A common offset cancels when we compare a pair. In the random-gap condition the full count recovered $117$ of the $122$ genuinely mixed pictures. The graph identity is exact under its hypotheses. Its application to pixels depends on measuring the three inputs faithfully.

### Back to the source drawings

We applied the same endpoint and region reader to the $88$ Rideout drawings without changing its parameters. Figure 7 shows a reversal. At $128$ pixels the full rule read $75$ diagrams correctly and the component counter read five. At $2{,}048$ pixels the component counter read all $88$, while the full rule got $55$ right. In all $33$ misses the component and region counts were already correct, but the thinned skeleton had one extra tip in $31$ cases and two extra tips in the other two. On Jkasd’s separate table the full rule read all fifteen entries at $48$ pixels and above, whereas the arc count got six of fifteen at $48$. Counting endpoints can help when gaps close, but it introduces a measurement that can fail when the arcs are already well separated.

[Vector illustration](../figures/poster-transfer.pdf)

Two pixel readers on the same <span class="math inline">88</span> vector diagrams rasterized at six resolutions. The knot at left is <span class="math inline">10<sub>3</sub></span> from Rideout’s poster [Rid20]. The underlying diagram is fixed throughout.

## What we would ask a knot detector

A correct count on a familiar table picture could come from counting double points, recognizing the knot type, or recalling the drawing. A curl separates the first two explanations. Holding out every relative of a drawing tests the third.

For a knot detector we would want the picture back, marked to show which strands meet and which one passes over. In Figure 1 one switch preserves the count of three and changes the number of tricolorings from nine to three. The worked trefoil in the accompanying appendix traces the strands and shows exactly which entry of the code changes.

##### Code and data.

The companion repository for the new experiments is [`dtubbenhauer/knotpictures`](https://github.com/dtubbenhauer/knotpictures) \[DKT26\]. The historical notebook \[Dra+25\] and the released images \[DKMT25\] are available at their original addresses.

##### Acknowledgments.

This project grew out of the Yulia’s Dream track of MIT PRIMES. We thank the organizers for their support, Radmila Sazdanović for pointing us to Robert Scharein’s work on knot visualization \[Sch22\], and Saul Schleimer for identifying the trefoil in our photograph before the first version appeared. D.T. acknowledges support from the Australian Research Council through Future Fellowship FT230100489. ChatGPT assisted with the experimental code, analysis, figures, and writing of this revision.

## References

- **[DKMT25]** A. Dranowski, Y. Kabkov, R. Melamud and D. Tubbenhauer, *Knot Crossing Dataset*, 2025. [Hugging Face dataset tr33hugg3r/knot-crossings](https://huggingface.co/datasets/tr33hugg3r/knot-crossings).
- **[DKT25]** A. Dranowski, Y. Kabkov and D. Tubbenhauer, *On knot detection via picture recognition*, original version, 2025. [arXiv:2510.06284v1](https://arxiv.org/abs/2510.06284v1).
- **[DKT26]** A. Dranowski, Y. Kabkov and D. Tubbenhauer. *Knotpictures. Companion code and data for On knot detection via picture recognition*, 2026. <https://github.com/dtubbenhauer/knotpictures>.
- **[Dra+25]** A. Dranowski and contributors, *Knot CNN*, historical generator and training notebook. [Latest.ipynb, commit 46fcbd90](https://github.com/annedranowski/knot-cnn/blob/46fcbd90a34b73775950090d91afe85d71e1f3db/notebooks/Latest.ipynb).
- **[GHRS21]** S. Gukov, J. Halverson, F. Ruehle, and P. Sułkowski. Learning to unknot. *Mach. Learn.: Sci. Technol.*, 2(2):025035, 2021. URL: <https://arxiv.org/abs/2010.16263>, [`doi:10.1088/2632-2153/abe91f`](https://doi.org/10.1088/2632-2153/abe91f).
- **[GJM+20]** R. Geirhos et al., Shortcut learning in deep neural networks, *Nature Machine Intelligence* **2** (2020), 665–673. [doi:10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z).
- **[Hu62]** M.-K. Hu, Visual pattern recognition by moment invariants, *IRE Trans. Information Theory* **8** (1962), 179–187. [doi:10.1109/TIT.1962.1057692](https://doi.org/10.1109/TIT.1962.1057692).
- **[Jka08]** Jkasd, *Knot table*, Wikimedia Commons, 2008, public domain. [File:Knot table.svg](https://commons.wikimedia.org/wiki/File:Knot_table.svg).
- **[LZ21]** H. Liu and H. Zhang, A suggestive interface for untangling mathematical knots, *IEEE Trans. Visualization and Computer Graphics* **27** (2021), 593–602. [doi:10.1109/TVCG.2020.3028893](https://doi.org/10.1109/TVCG.2020.3028893).
- **[Mil19]** K. Miller, *KnotFolio*, image importing and diagram conversion, 2019. [Project documentation](https://github.com/kmill/knotfolio/wiki).
- **[Ots79]** N. Otsu, A threshold selection method from gray-level histograms, *IEEE Trans. Systems, Man, and Cybernetics* **9** (1979), 62–66. [doi:10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076).
- **[Prz98]** J. H. Przytycki, 3-coloring and other elementary invariants of knots, *Banach Center Publications* **42** (1998), 275–295. [doi:10.4064/-42-1-275-295](https://doi.org/10.4064/-42-1-275-295).
- **[Rid20]** P. Rideout. Generating SVG for the Prime Knots. 2020. <https://prideout.net/blog/svg_knots/>.
- **[Sch22]** R.G. Scharein. KnotPlot. 2022. URL: <https://knotplot.com/>.
- **[ZS84]** T. Y. Zhang and C. Y. Suen, A fast parallel algorithm for thinning digital patterns, *Communications of the ACM* **27** (1984), 236–239. [doi:10.1145/357994.358023](https://doi.org/10.1145/357994.358023).
