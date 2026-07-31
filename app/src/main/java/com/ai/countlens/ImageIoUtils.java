package com.ai.countlens;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Image loading, full-resolution camera capture and gallery export utilities. */
final class ImageIoUtils {
    private static final String FILE_PROVIDER_SUFFIX = ".fileprovider";

    private ImageIoUtils() {
    }

    static Uri createCameraUri(Context context) throws IOException {
        File imageDirectory = new File(context.getCacheDir(), "images");
        if (!imageDirectory.exists() && !imageDirectory.mkdirs()) {
            throw new IOException("Could not create camera cache directory");
        }
        File imageFile = File.createTempFile("countlens_capture_", ".jpg", imageDirectory);
        return FileProvider.getUriForFile(
                context,
                context.getPackageName() + FILE_PROVIDER_SUFFIX,
                imageFile
        );
    }

    static Bitmap decodeBitmap(Context context, Uri uri, int maxSide) throws IOException {
        int safeMaxSide = Math.max(640, Math.min(4096, maxSide));
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ImageDecoder.Source source = ImageDecoder.createSource(context.getContentResolver(), uri);
            return ImageDecoder.decodeBitmap(source, (decoder, info, ignored) -> {
                android.util.Size size = info.getSize();
                int width = size.getWidth();
                int height = size.getHeight();
                int longest = Math.max(width, height);
                if (longest > safeMaxSide) {
                    float scale = safeMaxSide / (float) longest;
                    decoder.setTargetSize(
                            Math.max(1, Math.round(width * scale)),
                            Math.max(1, Math.round(height * scale))
                    );
                }
                decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                decoder.setMemorySizePolicy(ImageDecoder.MEMORY_POLICY_LOW_RAM);
            });
        }
        return decodeLegacy(context.getContentResolver(), uri, safeMaxSide);
    }

    @SuppressWarnings("deprecation")
    static Uri saveToGallery(Context context, Bitmap bitmap) throws IOException {
        String displayName = "CountLens_" + System.currentTimeMillis() + ".png";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentResolver resolver = context.getContentResolver();
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, displayName);
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
            values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/CountLens");
            values.put(MediaStore.Images.Media.IS_PENDING, 1);

            Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (uri == null) {
                throw new IOException("MediaStore could not create the destination image");
            }
            boolean completed = false;
            try (OutputStream stream = resolver.openOutputStream(uri, "w")) {
                if (stream == null || !bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                    throw new IOException("Could not encode the result image");
                }
                completed = true;
            } finally {
                if (completed) {
                    ContentValues publishValues = new ContentValues();
                    publishValues.put(MediaStore.Images.Media.IS_PENDING, 0);
                    resolver.update(uri, publishValues, null, null);
                } else {
                    resolver.delete(uri, null, null);
                }
            }
            return uri;
        }

        File pictures = new File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                "CountLens"
        );
        if (!pictures.exists() && !pictures.mkdirs()) {
            throw new IOException("Could not create the CountLens pictures directory");
        }
        File destination = new File(pictures, displayName);
        try (OutputStream stream = new FileOutputStream(destination)) {
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                throw new IOException("Could not encode the result image");
            }
        }
        MediaScannerConnection.scanFile(
                context,
                new String[]{destination.getAbsolutePath()},
                new String[]{"image/png"},
                null
        );
        return Uri.fromFile(destination);
    }

    private static Bitmap decodeLegacy(ContentResolver resolver, Uri uri, int maxSide) throws IOException {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream stream = resolver.openInputStream(uri)) {
            if (stream == null) {
                throw new IOException("Could not open the selected image");
            }
            BitmapFactory.decodeStream(stream, null, bounds);
        }

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inSampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight, maxSide);
        Bitmap bitmap;
        try (InputStream stream = resolver.openInputStream(uri)) {
            if (stream == null) {
                throw new IOException("Could not reopen the selected image");
            }
            bitmap = BitmapFactory.decodeStream(stream, null, options);
        }
        if (bitmap == null) {
            throw new IOException("The selected image could not be decoded");
        }

        int orientation = ExifInterface.ORIENTATION_NORMAL;
        try (InputStream stream = resolver.openInputStream(uri)) {
            if (stream != null) {
                ExifInterface exif = new ExifInterface(stream);
                orientation = exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                );
            }
        } catch (IOException ignored) {
            // The image is still usable even when metadata cannot be read.
        }
        return applyExifOrientation(bitmap, orientation);
    }

    private static int calculateSampleSize(int width, int height, int maxSide) {
        int sampleSize = 1;
        int longest = Math.max(width, height);
        while (longest > 0 && longest / (sampleSize * 2) >= maxSide) {
            sampleSize *= 2;
        }
        return sampleSize;
    }

    private static Bitmap applyExifOrientation(Bitmap bitmap, int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180f);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setScale(1f, -1f);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90f);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90f);
                break;
            default:
                return bitmap;
        }

        Bitmap oriented = Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.getWidth(),
                bitmap.getHeight(),
                matrix,
                true
        );
        if (oriented != bitmap) {
            bitmap.recycle();
        }
        return oriented;
    }
}
