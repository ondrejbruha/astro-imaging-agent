# Processing previews and quality measurements

`astroagent.preview.create_preview(image, pipeline, output, options=PreviewOptions(...),
context=...)` validates through the real registry and executes the same image tools
as full processing. The implemented strategy is `full-resolution-then-crop-resize`.
It processes the complete input first, then applies an optional zero-based
`(x, y, width, height)` ROI and a scale in `(0, 1]` to the display output.

This deliberately uses full-image global normalization, sky modeling and filter
neighborhoods; parameters such as Gaussian radius are not incorrectly reused on a
resized input. At scale one the processing values are equivalent to a crop of the
full result. The exported PNG separately maps finite ROI min/max to 8-bit values,
uses black for nonfinite samples and bilinear interpolation for reduced output.
Display mapping, quantization and resampling are recorded in `.preview.json` and
are not scientific processing steps. Constant regions map black. Original arrays,
headers and files are unchanged.

The provenance includes input identity/HDU, installed AIA version, resolved pipeline,
ROI/scale, strategy, equivalence/approximation and display mapping. Reduced displays
are approximate because of resampling; all PNGs are quantized display artifacts.
The maximum exported display is four million pixels by default, adjustable for direct
Python calls through validated `PreviewOptions`. Full-resolution computation may
still require substantial time/RAM. Accelerated ROI kernels, reduced-input processing
and persistent preview caching are unavailable. Registration/stacking are dataset
operations and do not advertise ordinary single-image previews; execute an explicitly
selected dataset pipeline to obtain a master and preview that master separately.

Worker preview generations increase per channel and supersede prior unsealed jobs.
Cancellation never publishes a complete result. A completion that raced a newer
request still carries its generation; the GUI must present only its current generation.
No display stretch is implicitly appended to a scientific pipeline.

## Half-flux radius

`HFRParams` and `measure_hfr(image, catalog, params)` are in `analysis.hfr`.
`DetectionParams.hfr` records the validated settings in analysis/registration reports
and resolved pipeline parameters. Detection reuses the existing DAOStarFinder catalog;
HFR is an aperture measurement of its detected centroids, independent of FWHM.

Defaults: aperture radius 8 pixels, sky annulus 10–14 pixels, and a 5×5 subpixel grid
within each pixel. Require `aperture_radius < background_inner < background_outer`.
The measurement uses zero-based DAO centroids at pixel centers and mono pixels or the
existing equal-weight RGB luminance. Sky is the median of finite annulus pixel centers.
Background-subtracted negative residuals are clipped to zero. Each pixel's flux is
uniformly divided between subpixel area samples, sorted by distance from the centroid.
Interpolate cumulative included flux at half of the flux inside the finite aperture.
The answer is a radius in pixels, not a diameter or a scaled FWHM proxy.

Sources with a truncated annulus, masked aperture/annulus, saturated aperture, nearby
catalog neighbor within twice the aperture radius, no positive aperture flux or
unresolved radius have null HFR and a per-source warning. Saturation checks the actual
input channels. Undetected neighbors and contaminated sky annuli can still bias the
measurement; it is not a crowded-field photometry solution. `DetectedStar.hfr` and
`hfr_warning` have defaults so old catalogs load. `FrameQualityMetrics.median_hfr`
and `StarMetrics.median_hfr` summarize reliable values only and default to null.
Missing/unavailable HFR never becomes zero. Other quality components remain usable.

For an untruncated circular Gaussian, analytic HFR is `sigma * sqrt(2 * ln(2))`.
Synthetic tests with sigma 1–2.5 pixels and subpixel centroids allow 0.13 pixel absolute
discretization error at defaults. Broad wings outside the finite aperture reduce HFR;
uniform pixel-area interpolation broadens undersampled stars. Negative-flux clipping
can bias low-SNR curves upward, and aperture/annulus choices affect extended objects.
Sky gradients, blends, saturation and bad masks limit reliability. This is an image
quality proxy, not calibrated photometry or aesthetic ranking.

HFR does not change the existing quality-score weights, reference-selection rules,
stacking weights or FWHM measurement. Batch analysis loads one frame at a time.
Session discovery stays header-only. New `FrameInfo` fields include raw `DATE-OBS`
capture time (no invented timezone), `INSTRUME`/`CAMERA`, `TELESCOP`, and projected
celestial WCS pixel scales in arcseconds per pixel. Missing values stay null, and
`metadata_sources` records actual header/WCS sources. WCS projected-plane scale is
an approximate local sampling measure, not an optical distortion calibration.
