import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.cluster import OPTICS

hdu = fits.open('./Fits_file/HOPS-182_subC18O.fits')
data = hdu[0].data.squeeze() 
snr_mask = data > 3 * np.std(data)

# Get indices of significant emission
z, y, x = np.where(snr_mask)
intensity = data[z, y, x]

# Features: (RA, Dec, Velocity, Intensity)
features = np.vstack([x, y, z, intensity]).T

# Apply OPTICS clustering
clust = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
clust.fit(features)

labels = clust.labels_

# Identify the main cluster
# (for example, keep the largest cluster)
from collections import Counter
counter = Counter(labels)
print(counter)

main_label = counter.most_common(1)[0][0]

# Create a clean mask
clean_mask = np.zeros_like(data, dtype=bool)
for i, lbl in enumerate(labels):
    if lbl == main_label:
        xi, yi, zi = x[i], y[i], z[i]
        clean_mask[zi, yi, xi] = True

# Apply mask
clean_data = np.zeros_like(data)
clean_data[clean_mask] = data[clean_mask]

# Save to new FITS
hdu[0].data = clean_data
hdu.writeto('HOPS-182_subC18O_cleaned.fits', overwrite=True)

# Quick look
plt.imshow(np.max(clean_data, axis=0), origin='lower')
plt.colorbar()
plt.title('Cleaned Streamer Emission (Max over Velocity)')
plt.show()
