/**
 * Tetris model
 * 
 * @author Jean-Charles Quinton
 * @version 07/06/2011
 * creation 07/06/2011
 */

public class Tetris extends CompoundLayer {
    /** Current tetromino */
    Tetromino tetromino;
    /** Fixed blocks */
    CompoundLayer fixed;

    /** Constructor for the Tetris model */
    public Tetris(int width, int height) {
        // Non movable pieces
        fixed = new CompoundLayer();
        add(fixed);
        // Add the border
        fixed.add(new Border(width,height));
        // Add the matrix of already stacked pieces
        // TODO fixed.add(new ...);
        // Moving tetromino
        tetromino = new Tetromino(0,height-3);
        add(tetromino);
    }
    
    /** Move the tetromino */
    public void moveTetromino(int dx) {
        // Generate a copy to test the movement
        Tetromino t = new Tetromino(tetromino);
        t.move(dx);
        // Test for overlaps around the tetromino
        // and only move it if there is room for it
        if (fixed.overlap(t,t.getBounds())==null)
            tetromino.move(dx);
    }
    
    /** Rotate the tetromino */
    public void rotateTetromino(int da) {
        // Generate a copy to test the movement
        Tetromino t = new Tetromino(tetromino);
        t.rotate(da);
        // Test for overlaps around the tetromino
        // and only move it if there is room for it
        if (fixed.overlap(t,t.getBounds())==null)
            tetromino.rotate(da);
    }
    
    /** Rotate the tetromino */
    public void fallTetromino() {
        // Generate a copy to test the movement
        Tetromino t = new Tetromino(tetromino);
        t.fall();
        // Test for overlaps around the tetromino
        // and only move it if there is room for it
        if (fixed.overlap(t,t.getBounds())==null)
            tetromino.fall();
    }
}