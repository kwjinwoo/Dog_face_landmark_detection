package com.example.tfliteinfer;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.example.tfliteinfer.tflite.ClassifierWithModel;

import java.io.File;
import java.io.IOException;

public class CameraActivity extends AppCompatActivity {
    public static final String TAG = "[IC]CameraActivity";
    public static final int CAMERA_IMAGE_REQUEST_CODE = 1;
    private static final String KEY_SELECTED_URI = "KEY_SELECTED_URI";

    private ClassifierWithModel cls;
    private ImageView imageView;
    private TextView textView;

    Uri selectedImageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        Button takeBtn = findViewById(R.id.takeBtn);
        takeBtn.setOnClickListener(v -> getImageFromCamera());

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);

        cls = new ClassifierWithModel(this);
        try {
            cls.init();
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        if(savedInstanceState != null) {
            Uri uri = savedInstanceState.getParcelable(KEY_SELECTED_URI);
            if (uri != null)
                selectedImageUri = uri;
        }
    }

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);

        outState.putParcelable(KEY_SELECTED_URI, selectedImageUri);
    }

    private void getImageFromCamera(){
        File file = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "picture.jpg");
        selectedImageUri = FileProvider.getUriForFile(this, getPackageName(), file);

        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, selectedImageUri);
        startActivityForResult(intent, CAMERA_IMAGE_REQUEST_CODE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK &&
                requestCode == CAMERA_IMAGE_REQUEST_CODE) {

            Bitmap bitmap = null;
            try {
                if(Build.VERSION.SDK_INT >= 29) {
                    ImageDecoder.Source src = ImageDecoder.createSource(
                            getContentResolver(), selectedImageUri);
                    bitmap = ImageDecoder.decodeBitmap(src);
                } else {
                    bitmap = MediaStore.Images.Media.getBitmap(
                            getContentResolver(), selectedImageUri);
                }
            } catch (IOException ioe) {
                Log.e(TAG, "Failed to read Image", ioe);
            }

            if(bitmap != null) {
                float[] output = cls.classify(bitmap);
//                String resultStr = String.format(Locale.ENGLISH,
//                        "class : %s, prob : %.2f%%",
//                        output.first, output.second * 100);

                imageView.setImageBitmap(bitmap);
//                textView.setText(output);
            }

        }
    }

    @Override
    protected void onDestroy() {
        cls.finish();
        super.onDestroy();
    }
}