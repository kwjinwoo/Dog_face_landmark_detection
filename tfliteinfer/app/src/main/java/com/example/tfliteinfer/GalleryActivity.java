package com.example.tfliteinfer;

import static java.lang.Integer.*;
import static java.sql.DriverManager.println;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
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
import android.widget.Toast;

import com.example.tfliteinfer.stickermaker.StickerMaker;
import com.example.tfliteinfer.tflite.ClassifierWithModel;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.text.SimpleDateFormat;

public class GalleryActivity extends AppCompatActivity implements View.OnClickListener {
    public static final String TAG = "[IC]GalleryActivity";
    public static final int GALLERY_IMAGE_REQUEST_CODE = 1;

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
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery);
        getImageFromGallery();
        Button selectBtn = findViewById(R.id.selectBtn);
        selectBtn.setOnClickListener(v -> getImageFromGallery());

        imageView = findViewById(R.id.imageView);
        backimageView = findViewById(R.id.backimageView);
        textView = findViewById(R.id.textView);
        ids = getResources().getStringArray(R.array.sticker_id);
        cls = new ClassifierWithModel(this);
        try {
            cls.init();
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

    }

    private void getImageFromGallery(){
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT).setType("image/*");
//        Intent intent = new Intent(Intent.ACTION_PICK,
//                MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(intent, GALLERY_IMAGE_REQUEST_CODE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK &&
                requestCode == GALLERY_IMAGE_REQUEST_CODE) {
            if (data == null) {
                return;
            }

            Uri selectedImage = data.getData();
            Bitmap bitmap = null;

            try {
                if(Build.VERSION.SDK_INT >= 29) {
                    ImageDecoder.Source src =
                            ImageDecoder.createSource(getContentResolver(), selectedImage);
                    bitmap = ImageDecoder.decodeBitmap(src);
                } else {
                    bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImage);
                }
            } catch (IOException ioe) {
                Log.e(TAG, "Failed to read Image", ioe);
            }

            if(bitmap != null) {
                output = cls.classify(bitmap); //모델 추론코드
                backimageView.setImageBitmap(bitmap);
//                내가 수정한 부분
//                Bitmap bitmap_canvas = Bitmap.createBitmap(imageView.getWidth(), imageView.getHeight(), Bitmap.Config.ARGB_8888);
//                Canvas tempCanvas = new Canvas(bitmap_canvas); //그림 넣을 캔버스 만들기

                int imageSize = 256;  // imageSize to rescale landmark

                //입력 이미지의 사이즈가 크기 때문에 이미지 뷰 영역에 맞춰줌
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
                tempCanvas = new Canvas(bitmap_canvas); //그림 넣을 캔버스 만들기
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap,(int) newWidth, (int) newHeight, false); //이미지 리사이징 실행코드
                Bitmap targetBmp = resizedBitmap.copy(Bitmap.Config.ARGB_8888, false); //위 비트맵 이미지를 그냥 넣으면 오류떠서 오류 해결코드
                tempCanvas.drawBitmap(targetBmp, 0, 0, null); //캔버스에 입력 이미지를 넣음
                paint = new Paint(Paint.ANTI_ALIAS_FLAG); //그림을 그릴 페인트 생성
//                paint.setColor(Color.CYAN); //점의 색 설정
                for (int index = 0; index <= 15;) { //점 찍는 반복문
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
//                for (int i = 0; i < ids.length; i++) {
//                    Rids.add(Integer.parseInt("R.id."+ids[i]));
//                    ImageButton btn = (ImageButton) findViewById(Rids.get(i));
//                    btn.setOnClickListener(this);
//                }
                skm = new StickerMaker();
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.bitsunglass);
                skm.make_sticker(tempCanvas, glasses, output, paint);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //입력이미지와 점을 이미지 뷰에 그려줌
//                textView.setText(Arrays.toString(output)); //모델 추론 결과값 확인을 위한 텍스트 출력
                Button savebtn = findViewById(R.id.savebtn);
                savebtn.setOnClickListener(v -> skm.saveBitmaptoJpeg(bitmap_canvas, this));
            }
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
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.railensunglass);
                tempCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                skm.make_sticker(tempCanvas, glasses, output, paint);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //입력이미지와 점을 이미지 뷰에 그려줌
                break;
            case R.id.bitsunglass:
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.bitsunglass);
                tempCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                skm.make_sticker(tempCanvas, glasses, output, paint);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //입력이미지와 점을 이미지 뷰에 그려줌
                break;
            case R.id.bdaysunglass:
                glasses = BitmapFactory.decodeResource(getApplicationContext().getResources(), R.drawable.bdaysunglass);
                tempCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
                skm.make_sticker(tempCanvas, glasses, output, paint);
                imageView.setImageDrawable(new BitmapDrawable(getResources(), bitmap_canvas)); //입력이미지와 점을 이미지 뷰에 그려줌
                break;
            default:
                break;
        }
    }
}