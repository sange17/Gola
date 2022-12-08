/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final int TF_OD_API_INPUT_SIZE = 416;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "yolov4-416.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/obj.txt";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.tfe_od_activity_camera);

        // 진료과 도착 완료
        Button arrivalBtn = findViewById(R.id.btn_next);
        arrivalBtn.setOnClickListener(e -> {

            // 다음 화면(메인 화면 Activity) 띄우기
            Intent intent = new Intent(this, MainActivity.class);
            startActivity(intent);
        });

    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    YoloV4Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED);
//            detector = TFLiteObjectDetectionAPIModel.create(
//                    getAssets(),
//                    TF_OD_API_MODEL_FILE,
//                    TF_OD_API_LABELS_FILE,
//                    TF_OD_API_INPUT_SIZE,
//                    TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

//        sunchip
//                corncho
//        swingchip
//                pocachip
//        caramelcornpeanut
//                shinjjang
//        saewookkang
//                honeybutterchip
//        cornchip
//                kkokkalcorn

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

                        // 객체 정보 출력(id, title, confidence, RectF(바운딩 박스 위치))
                        // ex) ==> [[10601] sunchip (77.1%) RectF(47.12654, 188.13676, 374.2074, 413.40082)]
                        Log.e("CHECK", "run: " + results);

                        // 스크린리더 사용 중 객체 클래스명 음성안내
                        try {
                            if(results.size() > 0){
                                String str = results.get(0).getTitle();
                                double confidence = results.get(0).getConfidence();
                                Toast toast;
                                Handler handler = new Handler();

                                if(confidence * 100 > 80){
                                    switch (str){
                                        case "sunchip":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "썬칩이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "corncho":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "콘초가 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "swingchip":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "스윙칩이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "pocachip":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "포카칩이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "caramelcornpeanut":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "카라멜콘땅콩이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "shinjjang":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "신짱이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "saewookkang":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "새우깡이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "honeybutterchip":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "허니버터칩이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "cornchip":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "콘칩이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                        case "kkokkalcorn":
                                            toast = Toast.makeText(
                                                    DetectorActivity.this,
                                                    "꼬깔콘이 인식되었습니다. 안내 음성이 계속되면 화면을 한 번 눌러주시고 과자 촬영을 종료하려면 화면을 두 번 눌러주세요.",
                                                    Toast.LENGTH_LONG);
                                            toast.show();
                                            break;
                                    }
                                }
                            }
                        } catch (final Exception e) {
                            LOGGER.e("Not Detection");
                        }



                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(5.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
