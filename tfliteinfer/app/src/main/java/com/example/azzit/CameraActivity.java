package com.example.azzit;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import com.example.azzit.stickermaker.StickerMaker;
import com.example.azzit.tflite.ClassifierWithModel;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CameraActivity extends AppCompatActivity implements View.OnClickListener {
    public static final String TAG = "[IC]CameraActivity";
    public static final int CAMERA_IMAGE_REQUEST_CODE = 1;
    private static final String KEY_SELECTED_URI = "KEY_SELECTED_URI";

    private ClassifierWithModel cls;
    private StickerMaker skm;
    private ImageView imageView;
    private TextView textView;
    private ImageView backimageView;
    public String[] ids;
    public List<Integer> Rids = new ArrayList<>();
    public Bitmap glasses;
    public float[] output;
    public Canvas tempCanvas;
    public Paint paint;
    public Bitmap bitmap_canvas;
    public Canvas stikerCanvas;
    public Bitmap targetBmp;


    Uri selectedImageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        getImageFromCamera();
        Button takeBtn = findViewById(R.id.takeBtn);
        takeBtn.setOnClickListener(v -> getImageFromCamera());
        findViewById(R.id.backbtn).setOnClickListener(v -> {
            Intent i = new Intent(getApplicationContext(), MainActivity.class);
            startActivity(i);
        });
        findViewById(R.id.sharebtn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bitmap_canvas != null) {
                    String bitmapPath = MediaStore.Images.Media.insertImage(getContentResolver(), bitmap_canvas, "title", null); //???????????? insert??????
                    Uri bitmapUri = Uri.parse(bitmapPath);//????????? ????????? Uri??? ????????????
                    Intent intent = new Intent(Intent.ACTION_SEND); //?????? ???????????? ?????????
                    intent.setType("image/*");//image?????????
                    intent.putExtra(Intent.EXTRA_STREAM, bitmapUri);
                    startActivity(Intent.createChooser(intent, "????????? ?????? ?????????"));
                }
            }
        });
        imageView = findViewById(R.id.imageView);
        Button savebtn = findViewById(R.id.savebtn);
        savebtn.setOnClickListener(v -> skm.saveBitmaptoJpeg(bitmap_canvas, this));

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
                output = cls.classify(bitmap);

//                int imageSize = 256;  // imageSize to rescale landmark
                int imageSize = 224;  // imageSize to rescale landmark


                //?????? ???????????? ???????????? ?????? ????????? ????????? ??? ????????? ?????????
                float newWidth = bitmap.getWidth();
                float newHeight = bitmap.getHeight();
                if (bitmap.getWidth() >= imageView.getWidth()) {
                    newWidth = imageView.getWidth();
                    float tempWidth = bitmap.getWidth();
                    float tempHeight = bitmap.getHeight();
                    newHeight = ((float)(newWidth / tempWidth))*tempHeight;
                    Log.d(TAG, "onActivityResult: "+ newHeight);
                    if (newHeight >= imageView.getHeight()) {
                        tempHeight = newHeight;
                        newHeight = imageView.getHeight();
                        newWidth = (newHeight/tempHeight)*newWidth;
                    }
                }
                bitmap_canvas = Bitmap.createBitmap((int) newWidth,(int) newHeight, Bitmap.Config.ARGB_8888);
                tempCanvas = new Canvas(bitmap_canvas); //?????? ?????? ????????? ?????????
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap,(int) newWidth, (int) newHeight, false); //????????? ???????????? ????????????
                targetBmp = resizedBitmap.copy(Bitmap.Config.ARGB_8888, false); //??? ????????? ???????????? ?????? ????????? ???????????? ?????? ????????????
                tempCanvas.drawBitmap(targetBmp, 0, 0, null); //???????????? ?????? ???????????? ??????
                paint = new Paint(Paint.ANTI_ALIAS_FLAG); //????????? ?????? ????????? ??????
//                paint.setColor(Color.CYAN); //?????? ??? ??????
                for (int index = 0; index <= 15;) { //??? ?????? ?????????
//                    tempCanvas.drawCircle(output[index]*newWidth/imageSize, output[index+1]*newHeight/imageSize, 8, paint);
                    output[index] = output[index]*newWidth/imageSize;
                    output[index+1] = output[index+1]*newHeight/imageSize;
                    index = index + 2;
                }
                ImageButton btn = (ImageButton) findViewById(R.id.railensunglass);
                btn.setOnClickListener(this);
                ImageButton btn2 = (ImageButton) findViewById(R.id.bitsunglass);
                btn2.setOnClickListener(this);
                ImageButton btn3 = (ImageButton) findViewById(R.id.bdaysunglass);
                btn3.setOnClickListener(this);
                ImageButton btn4 = (ImageButton) findViewById(R.id.aliensunglass);
                btn4.setOnClickListener(this);
                ImageButton btn5 = (ImageButton) findViewById(R.id.leonsunglass);
                btn5.setOnClickListener(this);

                skm = new StickerMaker();
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.bitsunglass);
                skm.make_sticker(tempCanvas, glasses, output, paint, 0.6);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas));

//                textView.setText(Arrays.toString(output)); //?????? ?????? ????????? ????????? ?????? ????????? ??????
//                   skm.saveBitmaptoJpeg(bitmap_canvas, this);
            }

        }
        else if (resultCode == RESULT_CANCELED){
            Intent intent = new Intent(this, MainActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP|Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
            CameraActivity.this.finish();
        }
    }

    @Override
    protected void onDestroy() {
        cls.finish();
        super.onDestroy();
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.railensunglass:
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.raliensunglass);
                tempCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                tempCanvas.drawBitmap(targetBmp, 0, 0, null); //???????????? ?????? ???????????? ??????
                skm.make_sticker(tempCanvas, glasses, output, paint, 1.6);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //?????????????????? ?????? ????????? ?????? ?????????
                break;
            case R.id.bitsunglass:
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.bitsunglass);
                tempCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                tempCanvas.drawBitmap(targetBmp, 0, 0, null); //???????????? ?????? ???????????? ??????
                skm.make_sticker(tempCanvas, glasses, output, paint, 0.6);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //?????????????????? ?????? ????????? ?????? ?????????
                break;
            case R.id.bdaysunglass:
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.bdaysunglass);
                tempCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                tempCanvas.drawBitmap(targetBmp, 0, 0, null); //???????????? ?????? ???????????? ??????
                skm.make_sticker(tempCanvas, glasses, output, paint,2.0);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //?????????????????? ?????? ????????? ?????? ?????????
                break;
            case R.id.aliensunglass:
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.aliensunglass);
                tempCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                tempCanvas.drawBitmap(targetBmp, 0, 0, null); //???????????? ?????? ???????????? ??????
                skm.make_sticker(tempCanvas, glasses, output, paint, 1.4);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //?????????????????? ?????? ????????? ?????? ?????????
                break;
            case R.id.leonsunglass:
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.leonsunglass);
                tempCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                tempCanvas.drawBitmap(targetBmp, 0, 0, null); //???????????? ?????? ???????????? ??????
                skm.make_sticker(tempCanvas, glasses, output, paint, 0.8);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //?????????????????? ?????? ????????? ?????? ?????????
                break;
            default:
                break;
        }
    }
}