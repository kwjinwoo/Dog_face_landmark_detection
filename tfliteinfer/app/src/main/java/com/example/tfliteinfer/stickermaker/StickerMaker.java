package com.example.tfliteinfer.stickermaker;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;

import com.example.tfliteinfer.R;

import org.tensorflow.lite.support.model.Model;

import java.io.IOException;

public class StickerMaker {
    public void init(float [] output) {
    }
//    private void heapter_glass() {
//
//    }
    public void make_sticker(Canvas tempCanvas, Bitmap sticker, float[] output, Paint paint, double height_ratio) {

        float left_x = output[14];
        float left_y = output[15];
        float right_x = output[6];
        float right_y = output[7];

//                float dist = (float) (right_x - left_x);    // naive distance
        float dist = (float) Math.sqrt((right_x-left_x)*(right_x-left_x)+(right_y-left_y)*(right_y-left_y));     // euclidean distance
        int width = (int) (dist*2.2);
        int height = (int) (dist*height_ratio);
//                int start_x = (int) (left_x+right_x)/2 - width/2;
//                int start_y = (int) (left_y+right_y)/2 - height/2;
        int start_x = (int) ((left_x+right_x)/2 - width/2);
        int start_y = (int) ((left_y+right_y)/2 - height/2);

//        paint.setColor(Color.RED);
//        tempCanvas.drawCircle(start_x, start_y, 8, paint);
//        tempCanvas.drawCircle(0, 0, 8, paint);
//        tempCanvas.drawCircle(left_x,left_y,8,paint);
//        tempCanvas.drawCircle(right_x,right_y,8,paint);
//        tempCanvas.drawCircle((left_x+right_x)/2,(left_y+right_y)/2, 8, paint);

        Matrix matrix = new Matrix();
        float rotation = (float) Math.atan2(right_y-left_y,right_x-left_x) * 180/(float)Math.PI; // rotation degree
        matrix.postRotate(rotation,(left_x+right_x)/2, (left_y+right_y)/2);

        Bitmap sticker2 = Bitmap.createScaledBitmap(sticker,width, height, false); //이미지 리사이징 실행코드

        Bitmap rotatedGlasses = Bitmap.createBitmap(sticker2, 0, 0, sticker2.getWidth(), sticker2.getHeight(), matrix, true);
        int w = Math.abs(rotatedGlasses.getWidth() - width);
        int h = Math.abs(rotatedGlasses.getHeight() - height);

        start_x -= (int) w / 2;
        start_y -= (int) h / 2;

//                Bitmap glasses3 = glasses2.copy(Bitmap.Config.ARGB_8888, false);
//                rotation
        tempCanvas.drawBitmap(rotatedGlasses, (int) start_x, (int) start_y, null);
//        return tempCanvas;
    }
}
