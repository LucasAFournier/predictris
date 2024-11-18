/**
 * Swing display of the Tetris blocks
 *
 * @author Jean-Charles Quinton
 * @version 07/06/11
 * creation 10/03/08
*/

import javax.swing.*;
import java.awt.*;
import java.awt.geom.*;
import java.awt.font.*;
import java.text.*;
import java.util.*;

import static java.lang.Math.*;

public class ViewSwing extends JPanelDB {    
    /** Associated view to display */
    View view;
    /** Sub view */
    View subview;
    
    /** Constructor */
    public ViewSwing(View view) {
        this.view = view;
    }
    
    @Override
    public void interact(EventObject e) {
        // Empty by default
    };
    
    @Override
    public void render(Graphics2D g) {
        g.clearRect(0,0,getWidth(),getHeight());
        int sx = getWidth()/view.width;
        int sy = getHeight()/view.height;
        sx = Math.min(sx,sy);
        sy = Math.min(sx,sy);
        // Iterate over the view
        for (int x=0; x<view.width; x++) {
            for (int y=0; y<view.height; y++) {
                if (view.blocks[x][y]!=Layer.NONE) {
                    g.setColor(Color.gray);
                    g.fill(new Rectangle2D.Double(
                        x*sx+1,(view.height-1-y)*sy+1,sx-2,sy-2
                    ));
                }
            }
        }
        // If a subview has been defined, display it */
        if (subview!=null) {
            g.setColor(Color.red);
            g.draw(new Rectangle2D.Double(
                (subview.x-view.x)*sx, ((view.height-subview.height)-(subview.y-view.y))*sy,
                subview.width*sx, subview.height*sy
            ));
        }
    }
}