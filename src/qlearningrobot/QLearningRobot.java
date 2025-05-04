package qlearningrobot;

import robocode.*;
import robocode.util.Utils;
import java.awt.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

public class QLearningRobot extends AdvancedRobot {

    // ----------------- HYPERPARAMETERS -----------------
    private static double  alpha        = 0.10;
    private static double  gamma        = 0.99;
    private static double  epsilon      = 0.30;
    private static double  alphaDecay   = 0.99995;
    private static double  epsilonDecay = 0.99995;
    private static double  alphaMin     = 0.01;
    private static double  epsilonMin   = 0.07;

    // ----------------- DISCRETIZATION -----------------
    private static final int B_ENEMY_X     = 7;
    private static final int B_ENEMY_Y     = 7;
    private static final int B_ANGLE       = 8;
    private static final int B_DISTANCE    = 6;
    private static final int B_ENEMY_FIRE  = 2;

    // --------------- ACTION SPACE ---------------------
    public static final int A_FIRE_LOW        = 0;
    public static final int A_FIRE_HIGH       = 1;
    public static final int A_AHEAD           = 2;
    public static final int A_BACK            = 3;
    public static final int A_TURN_LEFT       = 4;
    public static final int A_TURN_RIGHT      = 5;
    public static final int A_AHEAD_FIRE      = 6;
    public static final int A_BACK_FIRE       = 7;
    public static final int A_TURN_LEFT_FIRE  = 8;
    public static final int A_TURN_RIGHT_FIRE = 9;

    private static final int ACTIONS = 10;

    // --------------- REWARD PARAMS --------------------
    private static final double R_GOT_HIT          = -15.0;
    private static final double R_BULLET_HIT       =  10.0;
    private static final double R_BULLET_MISSED    =  -2.0;
    private static final double R_BULLET_HIT_BULLET=   1.0;
    private static final double R_HIT_WALL         =  -5.0;

    // -------------- FILE DATA ----------------------
    private final String QFILE      = "QTable.txt";
    private final String LOG_FILE   = "EpisodeRewards.txt";
    private final String HYPER_FILE = "Hyperparams.txt";
    private static final String PARAMS_FILE = "ParamsLog.txt";

    private static final String BASE_DIR =
            "C:\\Users\\Lenovo\\OneDrive\\Dokumenty\\semestr 8\\" +
                    "Symboliczne uczenie maszynowe\\iwisum-project\\"; //TODO change to your path


    private static final Map<Integer, double[]> qTable = new HashMap<>();
    private final Random rnd = new Random();

    private int lastState = -1;
    private int lastAction = -1;
    private static int episodeCount = 1;
    private static boolean started = true;
    private double cumulativeReward = 0.0;
    private double rewardSum = 0.0;

    private double lastEnemyEnergy = Double.NaN;


    @Override
    public void run() {
        checkDir();
        initColors();
        setAdjustRadarForRobotTurn(true);
        setAdjustGunForRobotTurn(true);
        ensureDirectoryExists();
        if (started) {
            loadQTable();
            loadHyperParams();
            loadEpisodeCount();
            started = false;
        }
        while (true) {
            turnRadarRight(360);
            waitFor(new RadarTurnCompleteCondition(this));
        }
    }

    //EVENTS
    @Override
    public void onScannedRobot(ScannedRobotEvent e) {
        int state = getState(e);

        if (lastState != -1 && lastAction != -1) {
            updateQ(cumulativeReward, lastState, state, lastAction);
            rewardSum += cumulativeReward;
            cumulativeReward = 0.0;
        }

        int action = chooseAction(state);
        performAction(e, action);
        lastState = state;
        lastAction = action;
        scan();
    }


    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        cumulativeReward += R_GOT_HIT * e.getPower();
    }

    @Override
    public void onBulletHit(BulletHitEvent e) {
        cumulativeReward += R_BULLET_HIT;
    }

    @Override
    public void onBulletMissed(BulletMissedEvent e) {
        cumulativeReward += R_BULLET_MISSED;
    }

    @Override
    public void onBulletHitBullet(BulletHitBulletEvent e) {
        cumulativeReward += R_BULLET_HIT_BULLET;
    }

    @Override
    public void onHitWall(HitWallEvent e) {
        cumulativeReward += R_HIT_WALL;
        setTurnRight(90 - e.getBearing());
        setAhead(100);
    }


    @Override
    public void onRoundEnded(RoundEndedEvent e) {
        saveEpisodeReward();
        decayParams();
        if (episodeCount % 100 == 0) {
            saveParams();
            saveHyperParams();
            saveQTable();
        }
        lastState = -1;
        lastAction = -1;
        cumulativeReward = 0.0;
        episodeCount++;
    }

    @Override
    public void onBattleEnded(BattleEndedEvent e) {
        saveQTable();
    }

    // ---------------------- DISCRETIZATION ---------------------------
    private int getState(ScannedRobotEvent e) {
        int enemyX = bucketEnemyX(e);
        int enemyY = bucketEnemyY(e);
        int angle  = bucketAngle(e);
        int dist   = bucketDistance(e.getDistance());
        int fired  = enemyFired(e) ? 1 : 0;

        int s = enemyX;
        s = s * B_ENEMY_Y    + enemyY;
        s = s * B_ANGLE      + angle;
        s = s * B_DISTANCE   + dist;
        s = s * B_ENEMY_FIRE + fired;
        return s;
    }

    private int bucketEnemyX(ScannedRobotEvent e) {
        double absBearing = getHeadingRadians() + e.getBearingRadians();
        double enemyX = getX() + e.getDistance() * Math.sin(absBearing);
        return (int) Math.min(B_ENEMY_X - 1, Math.max(0, enemyX / (getBattleFieldWidth() / B_ENEMY_X)));
    }

    private int bucketEnemyY(ScannedRobotEvent e) {
        double absBearing = getHeadingRadians() + e.getBearingRadians();
        double enemyY = getY() + e.getDistance() * Math.cos(absBearing);
        return (int) Math.min(B_ENEMY_Y - 1, Math.max(0, enemyY / (getBattleFieldHeight() / B_ENEMY_Y)));
    }

    private int bucketAngle(ScannedRobotEvent e) {
        int b = (int) ((e.getBearing() + 180.0) / (360.0 / B_ANGLE));
        return Math.max(0, Math.min(B_ANGLE - 1, b));
    }

    private int bucketDistance(double distance) {
        double span = 800.0 / B_DISTANCE;
        return (int) Math.min(B_DISTANCE - 1, distance / span);
    }

    private boolean enemyFired(ScannedRobotEvent e) {
        if (Double.isNaN(lastEnemyEnergy)) {
            lastEnemyEnergy = e.getEnergy();
            return false;
        }
        double delta = lastEnemyEnergy - e.getEnergy();
        lastEnemyEnergy = e.getEnergy();
        return delta >= 0.1 && delta <= 3.0;
    }

    // ---------------------- ACTIONS ---------------------------
    private int chooseAction(int state) {
        double[] qRow = qTable.get(state);
        if (qRow == null || rnd.nextDouble() < epsilon) {
            return rnd.nextInt(ACTIONS);
        }
        int best = 0;
        for (int i = 1; i < ACTIONS; i++) {
            if (qRow[i] > qRow[best]) best = i;
        }
        return best;
    }

    private void performAction(ScannedRobotEvent e, int action) {
        switch (action) {
            case A_FIRE_LOW:           predictiveShoot(e, 1.5);           break;
            case A_FIRE_HIGH:          predictiveShoot(e, 2.0);           break;
            case A_AHEAD:              setAhead(120);                     break;
            case A_BACK:               setBack(120);                      break;
            case A_TURN_LEFT:          setTurnLeft(45);  setAhead(50);    break;
            case A_TURN_RIGHT:         setTurnRight(45); setAhead(50);    break;
            case A_AHEAD_FIRE:         setAhead(120); predictiveShoot(e, 2.0); break;
            case A_BACK_FIRE:          setBack(120);  predictiveShoot(e, 2.0); break;
            case A_TURN_LEFT_FIRE:     setTurnLeft(45);  setAhead(50); predictiveShoot(e, 2.0); break;
            case A_TURN_RIGHT_FIRE:    setTurnRight(45); setAhead(50); predictiveShoot(e, 2.0); break;
        }
    }

    private void predictiveShoot(ScannedRobotEvent e, double power) {
        double bulletSpeed = 20 - 3 * power;
        double myX = getX(), myY = getY();
        double absBearing = getHeadingRadians() + e.getBearingRadians();
        double enemyX = myX + e.getDistance() * Math.sin(absBearing);
        double enemyY = myY + e.getDistance() * Math.cos(absBearing);
        double enemyHeading = e.getHeadingRadians();
        double enemyVelocity = e.getVelocity();

        double futureX = enemyX;
        double futureY = enemyY;
        double deltaTime = 0;
        while (deltaTime * bulletSpeed < Math.hypot(futureX - myX, futureY - myY)) {
            futureX += Math.sin(enemyHeading) * enemyVelocity;
            futureY += Math.cos(enemyHeading) * enemyVelocity;
            deltaTime += 1.0;
        }

        double aim = Math.atan2(futureX - myX, futureY - myY);
        double turn = Utils.normalRelativeAngle(aim - getGunHeadingRadians());
        turnGunRightRadians(turn);

        if (getGunHeat() == 0 && getEnergy() > power) {
            fire(power);
        }
    }

    // ---------------------- Q-LEARNING ---------------------------

    private void updateQ(double reward, int state, int newState, int action) {
        double[] qRow = qTable.computeIfAbsent(state, s -> new double[ACTIONS]);
        double[] qRowNew = qTable.computeIfAbsent(newState, s -> new double[ACTIONS]);

        double currentQ = qRow[action];
        double maxFuture = Arrays.stream(qRowNew).max().orElse(0.0);
        qRow[action] = (1 - alpha) * currentQ + alpha * (reward + gamma * maxFuture);
    }

    private void decayParams() {
        epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
        alpha   = Math.max(alphaMin,   alpha   * alphaDecay);
    }

    // ------------------- FILE METHODS -------------------
    private void checkDir() {
        File dataDir = getDataFile(".").getParentFile();
        if (!dataDir.exists()) dataDir.mkdirs();
    }

    private void initColors() {
        setBodyColor(Color.GRAY);
        setGunColor(Color.DARK_GRAY);
        setRadarColor(Color.LIGHT_GRAY);
        setBulletColor(Color.YELLOW);
    }

    private void ensureDirectoryExists() {
        File dir = new File(BASE_DIR);
        if (!dir.exists() && !dir.mkdirs()) {
            out.println("ERROR: Not Found " + BASE_DIR);
        }
    }

    private void loadQTable() {
        qTable.clear();
        String filePath = BASE_DIR + QFILE;
        File file = new File(filePath);
        if (!file.exists()) {
            out.println("QTable not found: " + filePath + " - creating new QTable");
            return;
        }
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(";");
                int state = Integer.parseInt(parts[0]);
                double[] actions = new double[ACTIONS];
                for (int i = 1; i <= ACTIONS && i < parts.length; i++) {
                    actions[i - 1] = Double.parseDouble(parts[i]);
                }
                qTable.put(state, actions);
            }
        } catch (Exception ex) {
            out.println("Error loading QTable: " + ex.getMessage());
        }
    }

    private void saveEpisodeReward() {
        String path = BASE_DIR + LOG_FILE;
        try (PrintStream ps = new PrintStream(new RobocodeFileOutputStream(path, true))) {
            ps.printf("%d;%.2f%n", episodeCount, rewardSum);
        } catch (IOException e) {
            out.println("Error saving episode reward: " + e.getMessage());
        }
        rewardSum = 0.0;
    }

    private void saveQTable() {
        String filePath = BASE_DIR + QFILE;
        try (PrintStream ps = new PrintStream(new RobocodeFileOutputStream(filePath))) {
            for (Map.Entry<Integer, double[]> entry : qTable.entrySet()) {
                StringJoiner sj = new StringJoiner(";");
                sj.add(entry.getKey().toString());
                for (double v : entry.getValue()) {
                    sj.add(Double.toString(v));
                }
                ps.println(sj);
            }
        } catch (IOException ex) {
            out.println("Error saving QTable: " + ex.getMessage());
        }
    }

    private void saveParams() {
        String path = BASE_DIR + PARAMS_FILE;
        try (PrintStream ps = new PrintStream(new RobocodeFileOutputStream(path, true))) {
            ps.printf(Locale.US,
                    "Episode:%d; alpha=%.5f; gamma=%.5f; epsilon=%.5f; alphaDecay=%.5f; epsilonDecay=%.5f; " +
                            "alphaMin=%.5f; epsilonMin=%.5f; B_ENEMY_X=%d; B_ENEMY_Y=%d; B_ANGLE=%d; B_DISTANCE=%d; B_FIRE=%d; STATES=%d; ACTIONS=%d\n",
                    episodeCount, alpha, gamma, epsilon, alphaDecay, epsilonDecay, alphaMin, epsilonMin,
                    B_ENEMY_X, B_ENEMY_Y, B_ANGLE, B_DISTANCE, B_ENEMY_FIRE, qTable.size(), ACTIONS);
        } catch (IOException ex) {
            out.println("Error saving params: " + ex.getMessage());
        }
    }

    private void loadHyperParams() {
        File file = new File(BASE_DIR + HYPER_FILE);
        if (!file.exists()) return;
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = br.readLine();
            if (line != null) {
                String[] p = line.split(";");
                alpha        = Double.parseDouble(p[0]);
                gamma        = Double.parseDouble(p[1]);
                epsilon      = Double.parseDouble(p[2]);
                alphaDecay   = Double.parseDouble(p[3]);
                epsilonDecay = Double.parseDouble(p[4]);
                alphaMin     = Double.parseDouble(p[5]);
                epsilonMin   = Double.parseDouble(p[6]);
            }
        } catch (Exception ex) {
            out.println("Error loading hyperparams: " + ex.getMessage());
        }
    }

    private void saveHyperParams() {
        File file = new File(BASE_DIR + HYPER_FILE);
        try (PrintStream ps = new PrintStream(new RobocodeFileOutputStream(file))) {
            ps.printf(Locale.US, "%.5f;%.5f;%.5f;%.5f;%.5f;%.5f;%.5f%n",
                    alpha, gamma, epsilon, alphaDecay, epsilonDecay, alphaMin, epsilonMin);
        } catch (IOException ex) {
            out.println("Error saving hyperparams: " + ex.getMessage());
        }
    }

    private void loadEpisodeCount() {
        Path path = Paths.get(BASE_DIR + LOG_FILE);
        if (!Files.exists(path)) {
            episodeCount = 1;
            return;
        }
        try {
            List<String> lines = Files.readAllLines(path);
            if (lines.isEmpty()) {
                episodeCount = 1;
            } else {
                String last = lines.get(lines.size() - 1);
                episodeCount = Integer.parseInt(last.split(";")[0]) + 1;
            }
        } catch (IOException | NumberFormatException ex) {
            out.println("Error loading episode count: " + ex.getMessage());
            episodeCount = 1;
        }
    }
}