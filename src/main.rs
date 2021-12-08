#![feature(drain_filter)]

use std::collections::BTreeMap;

use itertools::Itertools;

#[derive(PartialEq, Eq)]
enum Part {
    One,
    Two,
}

fn day1(part: Part) {
    let depths = include_str!("day1_input.txt");
    let depths: Vec<i32> = depths.lines().map(|line| line.parse().unwrap()).collect();

    match part {
        Part::One => {
            let n_larger = depths
                .windows(2)
                .filter(|slice| slice[1] > slice[0])
                .count();
            println!("{}", n_larger);
        }
        Part::Two => {
            let window_sums = depths
                .windows(3)
                .map(|window| -> i32 { window.iter().sum() });
            let n_larger = window_sums
                .tuple_windows()
                .filter(|(sum, next_sum)| next_sum > sum)
                .count();
            println!("{}", n_larger);
        }
    }
}

fn day2(part: Part) {
    let navigational_commands = include_str!("day2_input.txt");
    let mut horizontal_pos = 0;
    let mut depth = 0;
    let mut aim = 0;

    for command in navigational_commands.lines() {
        let (command, arg) = command.split_once(' ').unwrap();
        let arg: i64 = arg.parse().unwrap();

        match part {
            Part::One => match command {
                "forward" => horizontal_pos += arg,
                "down" => depth += arg,
                "up" => depth -= arg, // what happens if the depth turns out negative?
                _ => unreachable!(),
            },
            Part::Two => match command {
                "forward" => {
                    horizontal_pos += arg;
                    depth += aim * arg;
                }
                "down" => aim += arg,
                "up" => aim -= arg,
                _ => unreachable!(),
            },
        }
    }

    println!("{}", horizontal_pos * depth);
}

fn bit_iter(num: i64, n_len: usize) -> impl Iterator<Item = bool> {
    (0..n_len).map(move |pos| (1 << pos) & num != 0)
}

fn day3(part: Part) {
    let nums = include_str!("day3_input.txt");
    //let nums = include_str!("day3_test_input.txt");
    let n_width = nums.lines().next().unwrap().len();
    let nums = nums
        .lines()
        .map(|n| i64::from_str_radix(n, 2).unwrap())
        .collect::<Vec<_>>();

    let count_bit_occurences = |nums: &[i64]| {
        // 1 integer per bit position
        // add 1, if the bit is set for a number, subtract 1 if it's not set.
        // => >0 => more set than not set, -1 => opposite, 0 => equal
        // little endian
        let mut sum_for_bit = vec![0; n_width];

        for num in nums.iter() {
            for (bit_sum, bit) in sum_for_bit.iter_mut().zip(bit_iter(*num, n_width)) {
                *bit_sum += if bit { 1 } else { -1 };
            }
        }
        sum_for_bit
    };

    match part {
        Part::One => {
            let sum_for_bit = count_bit_occurences(&nums);
            let gamma = sum_for_bit.iter().rev().fold(0, |num, bit| {
                let most_common_bit = match bit.cmp(&0) {
                    std::cmp::Ordering::Less => 0,
                    std::cmp::Ordering::Equal => unreachable!(), // undefined state
                    std::cmp::Ordering::Greater => 1,
                };
                (num << 1) | most_common_bit
            });
            let epsilon = !gamma & ((1 << n_width) - 1);
            println!("{} * {} = {}", gamma, epsilon, gamma * epsilon);
        }
        Part::Two => {
            let find_rating = |filter_least_common: bool| {
                let mut candidates = nums.clone();

                let mut bit_pos = n_width - 1;

                while candidates.len() > 1 {
                    let sum_for_bit = count_bit_occurences(&candidates);
                    let most_common_bit = if ((sum_for_bit[bit_pos]) >= 0) ^ filter_least_common {
                        1
                    } else {
                        0
                    };
                    candidates.retain(|n| ((n >> bit_pos) & 1) == most_common_bit);
                    if bit_pos == 0 {
                        break;
                    }
                    bit_pos -= 1;
                }

                assert!(candidates.len() == 1);

                candidates[0]
            };
            let oxygen_rating = find_rating(false);
            let scrubber_rating = find_rating(true);
            let life_support_rating = oxygen_rating * scrubber_rating;
            println!(
                "{} * {} = {}",
                oxygen_rating, scrubber_rating, life_support_rating
            );
            println!(
                "binary: {:b} * {:b} = {:b}",
                oxygen_rating, scrubber_rating, life_support_rating
            );
        }
    }
}

fn parse_num(num: &str) -> i64 {
    num.parse().unwrap()
}

struct BingoBoard {
    nums: Vec<i64>,
    marked: [bool; 25],
}

impl BingoBoard {
    fn parse(board: &str) -> Self {
        Self {
            nums: board.split_whitespace().map(parse_num).collect(),
            marked: [false; 25],
        }
    }

    // mark field if the number is part of the board
    // and return the score of the board if it wins because of it
    fn mark_num(&mut self, num: i64) -> Option<i64> {
        if let Some((idx, _)) = self
            .nums
            .iter()
            .enumerate()
            .find(|(_, board_num)| num == **board_num)
        {
            self.marked[idx] = true;

            if self.is_bingo() {
                let sum_unmarked: i64 = self
                    .nums
                    .iter()
                    .zip(self.marked)
                    .filter(|(_, marked)| !*marked)
                    .map(|(num, _)| num)
                    .sum();
                return Some(num * sum_unmarked);
            }
        }
        None
    }

    fn is_bingo(&self) -> bool {
        (0..5).any(|row| (5 * row..5 * row + 5).all(|idx| self.marked[idx]))
            || (0..5).any(|col| (col..25).step_by(5).all(|idx| self.marked[idx]))
    }
}

fn day4(part: Part) {
    let input = include_str!("day4_input.txt");
    //let input = include_str!("day4_test_input.txt");
    let (all_nums, fields) = input.split_once("\n").unwrap();
    let all_nums = all_nums.split(',').map(parse_num).collect::<Vec<_>>();
    let mut boards = fields
        .split("\n\n")
        .map(BingoBoard::parse)
        .collect::<Vec<_>>();

    match part {
        Part::One => {
            'outer: for num in all_nums {
                for board in boards.iter_mut() {
                    if let Some(winning_score) = board.mark_num(num) {
                        println!("{}", winning_score);
                        break 'outer;
                    }
                }
            }
        }
        Part::Two => {
            // assuming all boards win eventually
            let mut nums = all_nums.iter();
            for &num in nums.by_ref() {
                // .retain() doesn't give us an &mut
                boards.drain_filter(|board| board.mark_num(num).is_some());
                if boards.len() == 1 {
                    break;
                }
            }
            let mut last_board = boards.into_iter().next().unwrap();
            let score = nums.find_map(|num| last_board.mark_num(*num)).unwrap();
            println!("{}", score);
        }
    }
}

struct Line {
    // (x, y)
    start: (i64, i64),
    end: (i64, i64),
}

impl Line {
    fn parse(line: &str) -> Self {
        let (start, end) = line.split_once(" -> ").unwrap();
        let parse_point = |nums: &str| {
            let (x, y) = nums.split_once(',').unwrap();
            (parse_num(x), parse_num(y))
        };
        Self {
            start: parse_point(start),
            end: parse_point(end),
        }
    }

    fn iter(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        let step_x = self.end.0.cmp(&self.start.0) as i64;
        let step_y = self.end.1.cmp(&self.start.1) as i64;

        let mut current_point = self.start;
        let mut last_reached = false;

        std::iter::from_fn(move || {
            if last_reached {
                return None;
            }

            let last_point = current_point;
            current_point.0 += step_x;
            current_point.1 += step_y;
            if last_point == self.end {
                last_reached = true;
            }
            Some(last_point)
        })
    }

    fn is_horizontal(&self) -> bool {
        self.start.1 == self.end.1
    }

    fn is_vertical(&self) -> bool {
        self.start.0 == self.end.0
    }
}

fn day5(part: Part) {
    let input = include_str!("day5_input.txt");
    //let input = include_str!("day5_test_input.txt");
    let mut lines = input.lines().map(Line::parse).collect::<Vec<_>>();

    // max_x + 1 = n_cols
    let get_xs = |line: &Line| [line.start.0, line.end.0];
    let n_cols = lines.iter().flat_map(get_xs).max().unwrap() + 1;
    let get_ys = |line: &Line| [line.start.1, line.end.1];
    let n_rows = lines.iter().flat_map(get_ys).max().unwrap() + 1;

    let mut map = vec![0; (n_cols * n_rows) as usize];

    if part == Part::One {
        lines.retain(|line| line.is_horizontal() || line.is_vertical());
    }

    let idx = |x, y| (y * n_cols + x) as usize;

    for line in lines {
        for (x, y) in line.iter() {
            map[idx(x, y)] += 1;
        }
    }

    // debug print the board
    // if true {
    //     for y in 0..n_rows {
    //         for x in 0..n_cols {
    //             let num = map[idx(x, y)];
    //             if num > 0 {
    //                 print!("{}", num);
    //             } else {
    //                 print!(".");
    //             }
    //         }
    //         println!();
    //     }
    // }

    println!("{}", map.iter().filter(|&&num| num > 1).count());
}

fn day6(part: Part) {
    let input = include_str!("day6_input.txt");
    let mut fish_population_by_age = [0u128; 9];
    for age in input.trim().split(',').map(parse_num) {
        fish_population_by_age[age as usize] += 1;
    }

    let n_days = match part {
        Part::One => 80,
        Part::Two => 256,
    };

    for _ in 0..n_days {
        fish_population_by_age.rotate_left(1);
        fish_population_by_age[6] += fish_population_by_age[8];
    }

    println!("{}", fish_population_by_age.iter().sum::<u128>());
}

fn day7(part: Part) {
    let input = include_str!("day7_input.txt");
    //let input = include_str!("day7_test_input.txt");

    let mut crab_positions = BTreeMap::new();
    for pos in input.trim().split(',').map(parse_num) {
        *crab_positions.entry(pos).or_insert(0) += 1;
    }

    match part {
        Part::One => {
            // minimze fuel => d/dx sum_i( |x_i - x| ) = 0. The abs function makes an algebraic solution difficult.
            // when moving the target position from one position to the next one to the right,
            // total fuel consumption rises by how many crabs are on or to the left of your previous position
            // and lowers by how many crabs are to the right.
            // Equilibrium is reached when equally many crabs are on both sides, then you can move either way
            // without changing cost.
            // Start on left-most position and move right until there aren't more crabs on the right anymore than the left.
            //
            // Alternative approach could be used with a simple sort:
            // The optimum will be on the median value for odd numbers or one of the neighbors of it
            // for even numbers. One would have to check both neighbors.

            let (pos, n_crabs_to_left) = crab_positions.iter().next().unwrap();
            let (mut pos, mut n_crabs_to_left) = (*pos, *n_crabs_to_left);
            let mut total_fuel: i64 = crab_positions
                .iter()
                .map(|(crab_pos, n_crabs)| (crab_pos - pos) * n_crabs)
                .sum();
            let mut n_crabs_to_right: i64 = crab_positions
                .iter()
                .skip(1)
                .map(|(_, n_crabs)| n_crabs)
                .sum();

            let mut crab_pos_iter = crab_positions.iter().skip(1);
            while n_crabs_to_right > n_crabs_to_left {
                let (&next_pos, &n_crabs) = crab_pos_iter.next().unwrap();
                let pos_diff = next_pos - pos;
                total_fuel += (n_crabs_to_left - n_crabs_to_right) * pos_diff;
                n_crabs_to_left += n_crabs;
                n_crabs_to_right -= n_crabs;
                pos = next_pos;
                if n_crabs_to_right <= n_crabs_to_left {
                    // there could be multiple equally good solutions
                    break;
                }
            }

            println!("best position: {}, fuel consumption: {}", pos, total_fuel);
        }
        Part::Two => {
            // The squaring actually makes algebra easier, because |x|² == x²
            // x == target_pos, x_i == initial position ob submarine nr. i
            // total_fuel_cost = sum_i fuel_cost_i(x) = sum_i ( [(x-x_i)² + |x-x_i|]/2 )
            // d/dx total_fuel_cost = 0  to find minimum
            // => d/dx total_fuel_cost = sum_i ( [2*(x-x_i) ± 1]/2  )
            //                         = sum_i ( (x-x_i) ± 1/2 )
            //                         = n * x - sum_i (x_i ± 1/2) = 0
            // => n*x = sum_i (x_i) + sum_i (± 1/2)
            // =>   x = sum_i (x_i)/n + sum_i (± 1/2) / n
            //
            // The first sum is the mean of all submarine positions. The second sum should be close to 0,
            // but is at most +1/2 and at least -1/2.
            // We need an integral solution so we have to check the neighbors of the result and find the minimum
            // among those.
            // Given the uncertainty of ± 1/2 in the result, we need to check not just the floor and the ceiling
            // but also the ceil of mean + 1 and the floor of mean - 1 because the ± 1/2 term could push the mean between
            // two different integers than where it starts.
            // Hypothesis: ceil() and floor() suffice in all cases. Dunno how I could prove that.
            let total_fuel = |target_pos: i64| -> i64 {
                crab_positions
                    .iter()
                    .map(|(crab_pos, n_crabs)| {
                        let diff = (crab_pos - target_pos).abs();
                        (diff * (diff + 1) / 2) * n_crabs
                    })
                    .sum()
            };

            // this would be easier without aggregating crab positions in the map
            let n_crabs_total: i64 = crab_positions.iter().map(|(_, n_crabs)| n_crabs).sum();
            let avg_pos = crab_positions
                .iter()
                .map(|(pos, n_crabs)| (pos * n_crabs) as f64)
                .sum::<f64>()
                / n_crabs_total as f64;

            let (pos, total_fuel) = (((avg_pos - 1.0).floor() as i64)
                ..=(avg_pos + 1.0).ceil() as i64)
                .map(|pos| (pos, total_fuel(pos)))
                .min_by_key(|&(_, total_fuel)| total_fuel)
                .unwrap();
            println!("best position: {}, fuel consumption: {}", pos, total_fuel);
        }
    };
}

type SegmentMask = u8;

fn day8(part: Part) {
    let input = include_str!("day8_input.txt");
    let segment_mask = |letter_mask: &str| {
        letter_mask
            .bytes()
            .map(|ch| ch - b'a')
            .map(|digit| 1 << digit)
            .fold(0u8, std::ops::BitOr::bitor)
    };
    let convert_masks = |letter_masks: &str| {
        letter_masks
            .split_whitespace()
            .map(segment_mask)
            .collect::<Vec<_>>()
    };
    let notes = input
        .lines()
        .map(|line| line.split_once(" | ").unwrap())
        .map(|(digit_masks, num)| (convert_masks(digit_masks), convert_masks(num)));

    // 1, 4, 7 and 8 have unique amounts of segments active and they are 2, 3, 4 or 7 (not in that order)
    let is_trivially_identifiable = |mask: SegmentMask| [2, 3, 4, 7].contains(&mask.count_ones());
    let solution = notes
        .into_iter()
        .flat_map(|(_, num_digits)| num_digits)
        .filter(|dig| is_trivially_identifiable(*dig))
        .count();
    println!("{}", solution);
}

fn main() {
    if false {
        day1(Part::One);
        day1(Part::Two);
        day2(Part::One);
        day2(Part::Two);
        day3(Part::One);
        day3(Part::Two);
        day4(Part::One);
        day4(Part::Two);
        day5(Part::One);
        day5(Part::Two);
        day6(Part::One);
        day6(Part::Two);
        day7(Part::One);
        day7(Part::Two);
    }
    day8(Part::One);
}
