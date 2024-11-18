/**
 * View of a layer
 * 
 * @author Jean-Charles Quinton
 * @version 07/06/2011
 * creation 07/06/2011
 */

import java.awt.*;

public class View extends Rectangle {
    /** Associated blocks */
    int[][] blocks;
    
    /** Constructor */
    public View(int x, int y, int width, int height) {
        super(x,y,width,height);
        // Initialize the block array
        blocks = new int[width][height];
    }
    
    public View(View ref) {
        this(ref.x,ref.y,ref.width,ref.height);
    }
    
    /** Obtain a view of the given layer (within this view coordinates)
     *  @param layer    layer from which block values should be extracted */
    public void get(Layer layer) {
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                blocks[x][y] = layer.get(this.x+x,this.y+y); 
            }
        }
    }
}