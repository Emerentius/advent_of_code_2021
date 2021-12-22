#![feature(drain_filter)]
#![feature(stmt_expr_attributes)]

use nom::IResult;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::{Add, RangeInclusive};
use std::str::FromStr;
use std::{
    cmp::{max, min, Reverse},
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
    ops::{Index, IndexMut},
};
use structopt::StructOpt;

use itertools::{izip, Itertools};

#[cfg(test)]
mod day18;

mod day19;

#[derive(PartialEq, Eq)]
pub enum Part {
    One,
    Two,
}

impl FromStr for Part {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "1" => Ok(Part::One),
            "2" => Ok(Part::Two),
            _ => Err("only part 1 and 2 exist".to_owned()),
        }
    }
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

fn mask_iter(mut mask: u8) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let lowest_bit = mask & (!mask + 1);
        let bit_pos = lowest_bit.trailing_zeros() as u8;
        mask ^= lowest_bit;
        Some(bit_pos)
    })
}

fn segment_mask(letter_mask: &str) -> SegmentMask {
    letter_mask
        .bytes()
        .map(|ch| ch - b'a')
        .map(|digit| 1 << digit)
        .fold(0u8, std::ops::BitOr::bitor)
}

fn digit_segment_masks() -> [SegmentMask; 10] {
    const DIGIT_SEGMENT_MASKS: [&str; 10] = [
        "abcefg", "cf", "acdeg", "acdfg", "bcdf", "abdfg", "abdefg", "acf", "abcdefg", "abcdfg",
    ];
    DIGIT_SEGMENT_MASKS.map(segment_mask)
}

// compute what the segment mask for a digit looks like with the jumbled wires given the deduced mapping
fn jumbled_segment_mask(digit_mask: u8, mapping: [u8; 7]) -> SegmentMask {
    mask_iter(digit_mask)
        .map(|desired_segment| mapping[desired_segment as usize])
        .fold(0, std::ops::BitOr::bitor)
}

fn day8(part: Part) {
    let input = include_str!("day8_input.txt");
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

    match part {
        Part::One => {
            // 1, 4, 7 and 8 have unique amounts of segments active and they are 2, 3, 4 or 7 (not in that order)
            let is_trivially_identifiable =
                |mask: SegmentMask| [2, 3, 4, 7].contains(&mask.count_ones());
            let solution = notes
                .into_iter()
                .flat_map(|(_, num_digits)| num_digits)
                .filter(|dig| is_trivially_identifiable(*dig))
                .count();
            println!("{}", solution);
        }
        Part::Two => {
            // Using bitmasks for two cases
            // 1. Which segment numbers are on in the output for a single digit.
            //    I use the type alias `SegmentMask` for this.
            //    segment positions are numbered 0 to 6 inclusive
            //    least significant bit == 0, most significant bit == 6
            // 2. Which segment in the output could be linked to a segment in the input.
            //    This is used in `day8_find_right_mapping()` and its result value `mapping`.
            //    That part could probably be completely abstracted inside `day8_find_right_mapping`.

            let mut sum = 0;
            for (digit_masks, num) in notes {
                let mapping = day8_find_right_mapping(&digit_masks);

                let mask_to_digit = digit_segment_masks()
                    .iter()
                    .enumerate()
                    .map(|(digit, mask)| (jumbled_segment_mask(*mask, mapping), digit))
                    .collect::<HashMap<_, _>>();
                let real_digits = num.iter().map(|digit| mask_to_digit[&digit]);
                sum += digits_to_num(real_digits);
            }

            println!("{}", sum);
        }
    }
}

fn digits_to_num(digits: impl IntoIterator<Item = usize>) -> usize {
    digits.into_iter().fold(0, |num, digit| num * 10 + digit)
}

fn day8_find_right_mapping(digit_masks: &[SegmentMask]) -> [u8; 7] {
    let mut possible_mappings = [0b_0111_1111u8; 7];

    for &mask in digit_masks {
        let desired_segments: &[_] = match mask.count_ones() {
            2 => &[2, 5],                // 1
            3 => &[0, 2, 5],             // 7
            4 => &[1, 2, 3, 5],          // 4
            7 => &[0, 1, 2, 3, 4, 5, 6], // 8
            _ => continue,
        };
        for segment in desired_segments {
            possible_mappings[*segment as usize] &= mask;
        }
    }

    let mut solutions = vec![];
    _day8_find_right_mapping(possible_mappings, [false; 7], &digit_masks, &mut solutions);
    assert!(solutions.len() == 1);
    solutions[0]
}

fn _day8_find_right_mapping(
    possible_mappings: [u8; 7],
    mut previously_selected_row: [bool; 7],
    numbers: &[SegmentMask],
    solutions: &mut Vec<[u8; 7]>,
) {
    let (row, (&min_poss_mask, _)) = match possible_mappings
        .iter()
        .zip(previously_selected_row)
        .enumerate()
        .filter(|(_, (_, done))| !done)
        .min_by_key(|(_, (map, _))| map.count_ones())
    {
        Some(res) => res,
        None => {
            // all rows already visited => the current mapping is possibly a solution. Check if we'd
            // get the right masks
            let jumbled_number_segment_masks =
                digit_segment_masks().map(|mask| jumbled_segment_mask(mask, possible_mappings));
            if numbers
                .iter()
                .all(|segment_mask| jumbled_number_segment_masks.contains(&segment_mask))
            {
                solutions.push(possible_mappings);
            }
            return;
        }
    };

    previously_selected_row[row] = true;

    for one_poss_col in mask_iter(min_poss_mask) {
        let chosen_possibility = 1 << one_poss_col;
        let mut new_possible_mappings = possible_mappings;
        new_possible_mappings[row] = chosen_possibility;
        for other_row in (0..7).filter(|&row2| row2 != row) {
            new_possible_mappings[other_row] &= !chosen_possibility;
        }

        _day8_find_right_mapping(
            new_possible_mappings,
            previously_selected_row,
            numbers,
            solutions,
        );
    }
}

fn neighbors(row: usize, col: usize) -> [(usize, usize); 4] {
    [
        (row.wrapping_sub(1), col), // up
        (row + 1, col),             // down
        (row, col.wrapping_sub(1)), // left
        (row, col + 1),             // right
    ]
}
fn day9(part: Part) {
    let input = include_str!("day9_input.txt");
    let height_map: Vec<Vec<_>> = input
        .lines()
        .map(|line| line.bytes().map(|byte| byte - b'0').collect())
        .collect();

    let minima = itertools::iproduct!(0..height_map.len(), 0..height_map[0].len())
        .map(|(r, c)| (r, c))
        .filter(|&(row, col)| {
            let height = height_map[row][col];
            neighbors(row, col)
                .into_iter()
                .filter_map(|(r, c)| height_map.get(r)?.get(c))
                .all(|&neighbor_height| neighbor_height > height)
        })
        .collect::<Vec<_>>();

    match part {
        Part::One => {
            let sum_risk_level = minima
                .into_iter()
                .map(|(r, c)| height_map[r][c] as u32 + 1)
                .sum::<u32>();
            println!("{}", sum_risk_level);
        }
        Part::Two => {
            let mut basins = minima
                .into_iter()
                .map(|(r, c)| find_cells_of_basin(r, c, &height_map))
                .collect::<Vec<_>>();
            basins.sort_by_key(HashSet::len);
            let n = basins.len();
            let solution: usize = basins[n - 3..].iter().rev().map(|hm| hm.len()).product();
            println!("{}", solution);
        }
    }
}

// Visit neighbors of the minimum and their neighbors, recursively
// in order of rising height (BFS).
// Every time we visit a new cell and all its neighbors that are not
// already part of the basin are higher, then the cell is also in the
// basin.
fn find_cells_of_basin(row: usize, col: usize, height_map: &[Vec<u8>]) -> HashSet<(usize, usize)> {
    let mut cells_to_check = BinaryHeap::new();
    cells_to_check.push((Reverse(height_map[row][col]), row, col));
    let mut already_visited = HashSet::new();
    already_visited.insert((row, col));

    let mut cells_of_basin = HashSet::new();

    while let Some((Reverse(height), row, col)) = cells_to_check.pop() {
        let new_neighbors = neighbors(row, col)
            .into_iter()
            .filter_map(|(r, c)| Some((r, c, *height_map.get(r)?.get(c)?)))
            .filter(|&(r, c, _)| !cells_of_basin.contains(&(r, c)))
            .filter(|&(_, _, height_n)| height_n < 9);

        if new_neighbors
            .clone()
            .all(|(_, _, height_n)| height_n >= height)
        {
            for (r, c, height) in new_neighbors {
                if already_visited.insert((r, c)) {
                    cells_to_check.push((Reverse(height), r, c));
                }
            }
            cells_of_basin.insert((row, col));
        }
    }
    cells_of_basin
}

fn day10(part: Part) {
    fn check_line(line: &str) -> Result<Vec<char>, char> {
        let mut required_closing_chars = vec![];

        for ch in line.chars() {
            let required_closer = match ch {
                '(' => ')',
                '{' => '}',
                '[' => ']',
                '<' => '>',
                // corrupt line, return first illegal character
                ')' | '>' | '}' | ']' => {
                    if required_closing_chars.pop() != Some(ch) {
                        return Err(ch);
                    }
                    continue;
                }
                _ => unreachable!(),
            };
            required_closing_chars.push(required_closer);
        }

        required_closing_chars.reverse();
        Ok(required_closing_chars)
    }

    let input = include_str!("day10_input.txt");

    match part {
        Part::One => {
            let solution = input
                .lines()
                .filter_map(|line| check_line(line).err())
                .map(|illegal_char| match illegal_char {
                    ')' => 3,
                    ']' => 57,
                    '}' => 1197,
                    '>' => 25137,
                    _ => unreachable!(),
                })
                .sum::<u64>();
            println!("{}", solution);
        }
        Part::Two => {
            let mut completion_scores = input
                .lines()
                .filter_map(|line| check_line(line).ok())
                .map(|required_closing_chars| {
                    required_closing_chars
                        .into_iter()
                        .map(|closing_ch| match closing_ch {
                            ')' => 1,
                            ']' => 2,
                            '}' => 3,
                            '>' => 4,
                            _ => unreachable!(),
                        })
                        .fold(0, |score, ch_score| score * 5 + ch_score)
                })
                .collect::<Vec<u64>>();
            completion_scores.sort();
            let mid = completion_scores.len() / 2;
            println!("{}", completion_scores[mid]);
        }
    }
}

fn day11(part: Part) {
    fn neighbors(cell: usize, n_rows: usize, n_cols: usize) -> impl Iterator<Item = usize> {
        let row = cell / n_cols;
        let col = cell % n_cols;

        let min_row = row.saturating_sub(1);
        let max_row = min(row + 1, n_rows - 1);
        let min_col = col.saturating_sub(1);
        let max_col = min(col + 1, n_cols - 1);
        itertools::iproduct!(min_row..=max_row, min_col..=max_col)
            .map(move |(r, c)| r * n_cols + c)
            .filter(move |&neighbor| neighbor != cell)
    }

    // returns how many flashes occured in the step
    fn simulate_step(energy_levels: &mut [i32], n_rows: usize, n_cols: usize) -> i32 {
        let mut n_flashes = 0;
        let mut any_octopus_flashes = false;

        for energy in energy_levels.iter_mut() {
            *energy += 1;
            any_octopus_flashes |= *energy == 10;
        }

        while any_octopus_flashes {
            any_octopus_flashes = false;

            for cell in 0..energy_levels.len() {
                if energy_levels[cell] >= 10 {
                    any_octopus_flashes = true;
                    n_flashes += 1;

                    energy_levels[cell] -= 30;
                    for neighbor_cell in neighbors(cell, n_rows, n_cols) {
                        energy_levels[neighbor_cell] += 1;
                    }
                }
            }
        }

        for energy in energy_levels {
            *energy = max(*energy, 0);
        }

        n_flashes
    }

    let input = include_str!("day11_input.txt");
    let mut energy_levels = input
        .lines()
        .flat_map(|line| line.bytes())
        .map(|byte| (byte - b'0') as i32)
        .collect::<Vec<_>>();

    let n_octopuses = energy_levels.len() as i32;
    let n_cols = input.lines().next().unwrap().len();
    let n_rows = energy_levels.len() / n_cols;

    let mut simulation_steps =
        (1..).map(|step| (step, simulate_step(&mut energy_levels, n_rows, n_cols)));

    match part {
        Part::One => {
            let n_flashes = simulation_steps
                .take(100)
                .map(|(_, n_flashes)| n_flashes)
                .sum::<i32>();
            println!("{}", n_flashes);
        }
        Part::Two => {
            let (synchronized_step, _) = simulation_steps
                .find(|&(_, n_flashes)| n_flashes == n_octopuses)
                .unwrap();
            println!("{}", synchronized_step);
        }
    }
}

fn day12_search_graph<'a>(
    current_cave: &'a str,
    connections: &HashMap<&'a str, HashSet<&'a str>>,
    path: &mut Vec<&'a str>,
    all_paths: &mut Vec<Vec<&'a str>>,
    n_visits_small_cave_max: usize,
) {
    if current_cave == "end" {
        all_paths.push(path.clone());
        return;
    }

    for &neighbor_cave in connections[current_cave].iter().filter(|&&c| c != "start") {
        let n_previous_visits = path.iter().filter(|&c| c == &neighbor_cave).count();
        let is_small_cave = neighbor_cave.chars().next().unwrap().is_lowercase();
        if is_small_cave && n_previous_visits >= n_visits_small_cave_max {
            continue;
        }
        let new_n_visits_max = if is_small_cave && n_previous_visits == 1 {
            1
        } else {
            n_visits_small_cave_max
        };

        path.push(neighbor_cave);
        day12_search_graph(
            neighbor_cave,
            connections,
            path,
            all_paths,
            new_n_visits_max,
        );
        path.pop();
    }
}

fn day12(part: Part) {
    let input = include_str!("day12_input.txt");
    let mut connections = HashMap::new();

    let insert_conn = |conns: &mut HashMap<_, _>, cave1, cave2| {
        conns.entry(cave1).or_insert(HashSet::new()).insert(cave2)
    };

    for connection in input.lines() {
        let (cave1, cave2) = connection.split_once("-").unwrap();
        insert_conn(&mut connections, cave1, cave2);
        insert_conn(&mut connections, cave2, cave1);
    }

    let n_visits_max = match part {
        Part::One => 1,
        Part::Two => 2,
    };

    let mut all_paths = vec![];
    day12_search_graph(
        "start",
        &connections,
        &mut vec!["start"],
        &mut all_paths,
        n_visits_max,
    );

    println!("{}", all_paths.len());
}

// generic 2D board
// This should come in handy with all these 2D-matrix problems
struct Board<T> {
    board: Vec<T>,
    n_cols: usize,
    n_rows: usize, // board.len() / n_cols
}

impl<T: Clone> Board<T> {
    fn new(default_val: T, n_rows: usize, n_cols: usize) -> Self {
        Board {
            board: vec![default_val; n_rows * n_cols],
            n_cols,
            n_rows,
        }
    }
}

impl<T> Board<T> {
    // map each char to an element
    fn from_str(board: &str, f: impl FnMut(char) -> T) -> Self {
        let n_cols = board.lines().next().unwrap().len();
        let n_rows = board.trim().len() / n_cols;
        let board = board
            .lines()
            .flat_map(|line| {
                assert!(line.len() == n_cols);
                line.chars()
            })
            .map(f)
            .collect_vec();

        Self {
            board,
            n_cols,
            n_rows,
        }
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn n_rows(&self) -> usize {
        self.board.len() / self.n_cols
    }

    fn cells(&self) -> impl Iterator<Item = (usize, usize)> {
        itertools::iproduct!(0..self.n_rows(), 0..self.n_cols()).map(|(y, x)| (x, y))
    }

    fn neighbors(&self, (row, col): (usize, usize)) -> impl Iterator<Item = (usize, usize)> {
        let n_rows = self.n_rows();
        let n_cols = self.n_cols();
        neighbors(row, col)
            .into_iter()
            .filter(move |&(row, col)| row < n_rows && col < n_cols)
    }

    fn get(&self, (x, y): (usize, usize)) -> Option<&T> {
        if x >= self.n_cols || y >= self.n_rows {
            return None;
        }
        self.board.get(y * self.n_cols + x)
    }
}

impl<T> Index<(usize, usize)> for Board<T> {
    type Output = T;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        assert!(x < self.n_cols);
        &self.board[y * self.n_cols + x]
    }
}

impl<T> IndexMut<(usize, usize)> for Board<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        assert!(x < self.n_cols);
        &mut self.board[y * self.n_cols + x]
    }
}

impl Display for Board<bool> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.n_rows() {
            for x in 0..self.n_cols() {
                let print_ch = if self[(x, y)] { '#' } else { '.' };
                write!(f, "{}", print_ch)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

fn day13(part: Part) {
    fn fold_coordinate(pos: usize, fold_coordinate: usize) -> Option<usize> {
        (pos != fold_coordinate).then(|| {
            if pos > fold_coordinate {
                2 * fold_coordinate - pos
            } else {
                pos
            }
        })
    }

    let input = include_str!("day13_input.txt");
    let (dots, folds) = input.split_once("\n\n").unwrap();
    let dots = dots
        .lines()
        .map(|line| line.split_once(',').unwrap())
        .map(|(x, y)| (parse_num(x) as usize, parse_num(y) as usize))
        .collect_vec();
    let max_x = dots.iter().map(|&(x, _)| x).max().unwrap();
    let max_y = dots.iter().map(|&(_, y)| y).max().unwrap();

    let mut board = Board::new(false, max_y + 1, max_x + 1);
    for (x, y) in dots {
        board[(x, y)] = true;
    }

    let folds = folds
        .lines()
        .map(|line| {
            line.strip_prefix("fold along ")
                .unwrap()
                .split_once("=")
                .unwrap()
        })
        .map(|(direction, coordinate)| (direction, parse_num(coordinate) as usize))
        .collect_vec();

    fn apply_fold(board: Board<bool>, fold_direction: &str, coordinate: usize) -> Board<bool> {
        let (new_n_cols, new_n_rows, new_coordinate) = match fold_direction {
            "x" => {
                assert_eq!(coordinate * 2 + 1, board.n_cols());
                let new_coord: Box<dyn Fn(_, _) -> _> =
                    Box::new(|x, y| Some((fold_coordinate(x, coordinate)?, y)));
                (board.n_cols() / 2, board.n_rows(), new_coord)
            }
            "y" => {
                assert_eq!(coordinate * 2 + 1, board.n_rows());
                let new_coord = Box::new(|x, y| Some((x, fold_coordinate(y, coordinate)?))) as _;
                (board.n_cols(), board.n_rows() / 2, new_coord)
            }
            _ => unreachable!(),
        };

        let mut new_board = Board::new(false, new_n_rows, new_n_cols);
        for (x, y) in board.cells() {
            if let Some(new_coords) = new_coordinate(x, y) {
                new_board[new_coords] |= board[(x, y)];
            }
        }

        new_board
    }

    match part {
        Part::One => {
            let (fold_direction, coordinate) = folds[0];
            let new_board = apply_fold(board, fold_direction, coordinate);
            println!(
                "{}",
                new_board.board.iter().filter(|&&marked| marked).count()
            );
        }
        Part::Two => {
            let final_board = folds.into_iter().fold(board, |board, (direction, coord)| {
                apply_fold(board, direction, coord)
            });
            for row in 0..final_board.n_rows() {
                for col in 0..final_board.n_cols() {
                    print!("{0}{0}", if final_board[(col, row)] { '█' } else { ' ' });
                }
                println!();
            }
        }
    }
}

fn count_occurences<T: Eq + Hash>(iter: impl Iterator<Item = T>) -> HashMap<T, usize> {
    sum_occurences(iter.map(|item| (item, 1)))
}

fn sum_occurences<T: Eq + Hash>(iter: impl Iterator<Item = (T, usize)>) -> HashMap<T, usize> {
    let mut count = HashMap::new();
    for (item, occ) in iter {
        *count.entry(item).or_insert(0) += occ;
    }
    count
}

fn day14(part: Part) {
    let input = include_str!("day14_input.txt");
    let (template, rules) = input.split_once("\n\n").unwrap();
    let template = template.as_bytes();
    let rules = rules
        .lines()
        .map(|line| line.split_once(" -> ").unwrap())
        .map(|(pair, insertion)| {
            let pair = pair.as_bytes();
            ((pair[0], pair[1]), insertion.as_bytes()[0])
        })
        .collect::<HashMap<_, _>>();

    let pairs = template.windows(2).map(|p| (p[0], p[1]));
    let pair_counts = count_occurences(pairs);
    let next_pair_counts = |pair_counts: HashMap<(u8, u8), usize>| {
        sum_occurences(pair_counts.into_iter().flat_map(|(pair, count)| {
            // if an insertion rule exists, map the pair onto two new pairs and sum their counts
            if let Some(&insertion) = rules.get(&pair) {
                [(pair.0, insertion), (insertion, pair.1)].map(|p| (p, count))
            } else {
                // keep just the one pair. The second pair of count 0 is to have a uniform return type.
                [(pair, count), (pair, 0)]
            }
        }))
    };

    let n_repetitions = match part {
        Part::One => 10,
        Part::Two => 40,
    };
    let pair_counts =
        (0..n_repetitions).fold(pair_counts, |pair_counts, _| next_pair_counts(pair_counts));
    // every character is part of two pairs except for the very first and last
    // so count only the first character of each pair which counts every char once
    // except for the last one and then add that one after.
    let mut char_counts = sum_occurences(pair_counts.into_iter().map(|((c1, _), occ)| (c1, occ)));
    *char_counts.get_mut(template.last().unwrap()).unwrap() += 1;

    let min_count = char_counts.values().min().unwrap();
    let max_count = char_counts.values().max().unwrap();

    println!("{}", max_count - min_count);
}

fn day15(part: Part) {
    let input = include_str!("day15_input.txt");
    let n_cols = input.lines().next().unwrap().len();
    let n_rows = input.trim().len() / n_cols;

    let board = {
        let mut base_board = Board::new(0, n_rows, n_cols);
        for (risk, cell) in input
            .lines()
            .flat_map(|line| line.bytes())
            .map(|risk| (risk - b'0') as i32)
            .zip(base_board.cells())
        {
            base_board[cell] = risk;
        }

        match part {
            Part::One => base_board,
            Part::Two => {
                let mut total_board = Board::new(0, n_rows * 5, n_cols * 5);
                for (row, col) in total_board.cells() {
                    let tile_row = (row / n_rows) as i32;
                    let tile_col = (col / n_cols) as i32;
                    let risk = base_board[(row % n_rows, col % n_cols)] + tile_row + tile_col;
                    let risk = if risk > 9 { risk - 9 } else { risk };
                    total_board[(row, col)] = risk;
                }
                total_board
            }
        }
    };
    let dest = (board.n_cols() - 1, board.n_rows() - 1);

    // Dijkstra's algorithm
    let mut already_visited = HashSet::new();
    let mut next_node = BinaryHeap::new();

    already_visited.insert((0, 0));
    next_node.push((Reverse(0), (0, 0)));

    let total_risk = loop {
        let (Reverse(path_len), cell) = next_node.pop().unwrap();
        if cell == dest {
            break path_len;
        }

        for neighbor in board.neighbors(cell) {
            if already_visited.insert(neighbor) {
                next_node.push((Reverse(path_len + board[neighbor]), neighbor));
            }
        }
    };

    println!("{}", total_risk);
}

#[derive(Debug, Eq, PartialEq)]
enum PacketContent {
    Literal(u64),
    Operator {
        type_id: u8,
        subpackets: Vec<Packet>,
    },
}

fn be_bits_to_num(bits: &[bool]) -> u64 {
    bits.iter().fold(0, |num, &bit| (num << 1) + bit as u64)
}

impl PacketContent {
    fn parse(mut bits: &[bool]) -> Result<(Self, &[bool]), ()> {
        let packet_type_id = be_bits_to_num(&bits[0..3]) as u8;
        bits = &bits[3..];
        let content = match packet_type_id {
            4 => {
                let mut num = 0;
                for (limb_nr, limb) in bits.chunks_exact(5).enumerate() {
                    num = (num << 4) + be_bits_to_num(&limb[1..5]);

                    if !limb[0] {
                        bits = &bits[(limb_nr + 1) * 5..];
                        break;
                    }
                    assert!(
                        limb_nr * 4 < 64,
                        "Literal value requires a bigger output integer type"
                    );
                }
                PacketContent::Literal(num)
            }
            _ => {
                let length_type_id = bits[0]; // 0 == false, 1 == true

                let mut subpackets = vec![];
                if length_type_id {
                    let n_subpackets = be_bits_to_num(&bits[1..12]);
                    bits = &bits[12..];
                    for _ in 0..n_subpackets {
                        let (packet, remaining_input) = Packet::parse(&bits)?;
                        bits = remaining_input;
                        subpackets.push(packet);
                    }
                } else {
                    let n_total_subpacket_length_in_bits = be_bits_to_num(&bits[1..16]);
                    bits = &bits[16..];
                    let mut n_length_subpackets_so_far = 0;
                    while n_length_subpackets_so_far < n_total_subpacket_length_in_bits {
                        let (packet, remaining_input) = Packet::parse(&bits)?;
                        n_length_subpackets_so_far += (bits.len() - remaining_input.len()) as u64;
                        bits = remaining_input;
                        subpackets.push(packet);
                    }
                }
                PacketContent::Operator {
                    type_id: packet_type_id,
                    subpackets,
                }
            }
        };

        Ok((content, bits))
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Packet {
    version: u8,
    content: PacketContent,
}

impl Packet {
    fn parse_str(input: &str) -> Result<Self, ()> {
        let bit_input = hexadecimal_to_bit_vec(input);
        Self::parse(&bit_input).map(|(packet, _remaining_input)| {
            assert!(_remaining_input.iter().all(|&bit| bit == false));
            packet
        })
    }

    fn parse(bits: &[bool]) -> Result<(Self, &[bool]), ()> {
        let version = bits[..3].iter().fold(0, |num, &bit| (num << 1) + bit as u8);
        let (content, remaining_input) = PacketContent::parse(&bits[3..])?;
        Ok((Self { version, content }, remaining_input))
    }

    fn value(&self) -> u64 {
        match &self.content {
            &PacketContent::Literal(val) => val,
            PacketContent::Operator {
                type_id,
                subpackets,
            } => {
                let sub_vals = subpackets.iter().map(Packet::value);
                match type_id {
                    0 => sub_vals.sum(),
                    1 => sub_vals.product(),
                    2 => sub_vals.min().unwrap(),
                    3 => sub_vals.max().unwrap(),
                    5 => (subpackets[0].value() > subpackets[1].value()) as u64,
                    6 => (subpackets[0].value() < subpackets[1].value()) as u64,
                    7 => (subpackets[0].value() == subpackets[1].value()) as u64,
                    _ => unreachable!(),
                }
            }
        }
    }
}

fn hexadecimal_to_bit_vec(input: &str) -> Vec<bool> {
    let bit_vec = input
        .trim()
        .as_bytes()
        .iter()
        .map(|&b| match b {
            b'0'..=b'9' => b - b'0',
            b'A'..=b'F' => b - b'A' + 10,
            _ => unreachable!(),
        })
        .flat_map(|byte| {
            (0..4)
                .rev()
                .map(move |bit_pos| ((byte >> bit_pos) & 1) != 0)
        })
        .collect_vec();
    bit_vec
}

fn day16(part: Part) {
    let input = include_str!("day16_input.txt");
    let packet = Packet::parse_str(input).unwrap();

    match part {
        Part::One => {
            fn version_sum(packet: Packet) -> u64 {
                let subpacket_version_sum = match packet.content {
                    PacketContent::Literal(_) => 0,
                    PacketContent::Operator { subpackets, .. } => {
                        subpackets.into_iter().map(version_sum).sum::<u64>()
                    }
                };
                packet.version as u64 + subpacket_version_sum
            }
            println!("{}", version_sum(packet));
        }
        Part::Two => {
            println!("{}", packet.value());
        }
    }
}

#[test]
fn test_day16_parse_literal() {
    assert_eq!(
        Packet::parse_str("D2FE28").unwrap(),
        Packet {
            version: 6,
            content: PacketContent::Literal(2021)
        }
    );
}

#[test]
fn test_day16_parse_simple_operator() {
    assert_eq!(
        Packet::parse_str("38006F45291200").unwrap(),
        Packet {
            version: 1,
            content: PacketContent::Operator {
                type_id: 6,
                subpackets: vec![
                    Packet {
                        version: 6,
                        content: PacketContent::Literal(10)
                    },
                    Packet {
                        version: 2,
                        content: PacketContent::Literal(20)
                    }
                ]
            }
        }
    );
}

#[test]
fn test_hexadecimal_to_bits() {
    assert_eq!(
        hexadecimal_to_bit_vec("D2FE28"),
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,]
            .map(|b| b == 1)
            .to_owned(),
    );
}

fn day17(part: Part) {
    let input = include_str!("day17_input.txt");
    let (_, x_min, x_max, y_min, y_max) = lazy_regex::regex_captures!(
        r"target area: x=(-?\d+)\.\.(-?\d+), y=(-?\d+)\.\.(-?\d+)",
        input.trim()
    )
    .unwrap();
    let x_min = parse_num(x_min);
    let x_max = parse_num(x_max);
    let y_min = parse_num(y_min);
    let y_max = parse_num(y_max);

    match part {
        Part::One => {
            // with a positive velocity, the probe will first climb up to vy * (vy + 1) / 2
            // linger for one step while its velocity goes to -1
            // and then it falls back down, visiting all the same heights but with vy = -vy_climb(y) - 1.
            // It will also reach y = 0 again.
            // At y = 0, the maximum speed in negative direction that doesn't overshoot the area is vy = y_min.
            let y_vel_max = -(y_min + 1);
            let y_max = y_vel_max * (y_vel_max + 1) / 2;

            println!("{}", y_max);
        }
        Part::Two => {
            let y_probe_positions = |y_vel_initial| {
                std::iter::successors(Some((0, y_vel_initial)), |(y, vy)| Some((y + vy, vy - 1)))
            };
            let x_probe_positions = |x_vel_initial: i64| {
                std::iter::successors(Some((0, x_vel_initial)), |(x, vx)| {
                    Some((x + vx, vx - vx.signum()))
                })
            };

            // x and y positions and velocities act independently
            // Per dimension, find velocities that could end up in the target strip.
            // Filter the combination of those for velocities that actually do.
            let potential_initial_vy = y_min..=-(y_min + 1);
            let will_end_in_xrange = |vx| {
                x_probe_positions(vx)
                    .take_while(|&(_, vx)| vx > 0)
                    .take_while(|&(x, _)| x <= x_max)
                    .any(|(x, _)| (x_min..=x_max).contains(&x))
            };
            let potential_initial_vx = (1..=x_max).filter(|&vx| will_end_in_xrange(vx));

            let solution = itertools::iproduct!(potential_initial_vx, potential_initial_vy)
                .filter(|&(vx, vy)| {
                    x_probe_positions(vx)
                        .zip(y_probe_positions(vy))
                        .take_while(|&((x, vx), (y, _))| {
                            x <= x_max && (x >= x_min || vx != 0) && y >= y_min
                        })
                        .any(|((x, _), (y, _))| {
                            (y_min..=y_max).contains(&y) && (x_min..=x_max).contains(&x)
                        })
                })
                .count();

            println!("{}", solution);
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum NumberNode {
    Number(i64),
    Pair(Box<(NumberNode, NumberNode)>),
}

impl NumberNode {
    fn magnitude(&self) -> i64 {
        match self {
            &NumberNode::Number(num) => num,
            NumberNode::Pair(pair) => {
                let (left, right) = &**pair;
                3 * left.magnitude() + 2 * right.magnitude()
            }
        }
    }
}

impl FromStr for SnailfishNumber {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        fn parse_pair(i: &str) -> IResult<&str, NumberNode> {
            use nom::character::complete::*;
            let (i, _) = char('[')(i)?;
            let (i, left) = parse_number_node(i)?;
            let (i, _) = char(',')(i)?;
            let (i, right) = parse_number_node(i)?;
            let (i, _) = char(']')(i)?;
            Ok((i, NumberNode::Pair(Box::new((left, right)))))
        }

        fn parse_number_node(i: &str) -> IResult<&str, NumberNode> {
            use nom::combinator::map;
            let parse_number = map(nom::character::complete::i64, NumberNode::Number);
            nom::branch::alt((parse_number, parse_pair))(i)
        }

        let (_, pair) = parse_pair(s).map_err(|e| e.to_owned())?;
        Ok(SnailfishNumber { root: pair })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Position {
    Left,
    Mid,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SnailfishNumber {
    // Should be a pair, but a user of the cursor
    // could violate it by replacing the root node with a number.
    root: NumberNode,
}

impl SnailfishNumber {
    fn explode_nums(&mut self) {
        use NumberNode::*;
        let mut cursor = self.cursor_start();
        while let Some((elem, depth)) = cursor.next() {
            match elem {
                NumberNode::Pair(pair) if depth >= 4 => {
                    let (left, right) = match &**pair {
                        &(Number(left), Number(right)) => (left, right),
                        _ => unreachable!(),
                    };
                    *elem = NumberNode::Number(0);
                    if let Some(next) = cursor.borrowing_clone().next_num() {
                        *next += right;
                    }
                    if let Some(prev) = cursor.borrowing_clone().prev_num() {
                        *prev += left;
                    }
                }
                _ => {}
            }
        }
    }

    // returns true if explosions are needed
    fn split_nums(&mut self) -> bool {
        use NumberNode::*;
        let mut cursor = self.cursor_start();
        while let Some((elem, depth)) = cursor.next() {
            match elem {
                &mut NumberNode::Number(num) if num >= 10 => {
                    let left = num / 2;
                    let right = num - left;
                    *elem = Pair(Box::new((Number(left), Number(right))));

                    // After a number is split, the result may have to be exploded or split again.
                    // Just `return true` would work. This is somewhat optimized already to avoid
                    // a lot of unnecessary iteration, but it could be optimized a lot more.
                    // The explosion could be done immediately. The cursor would need to be backtracked
                    // to the last num to check if it needs to be split.
                    if depth >= 4 {
                        // need to explode stuff again
                        return true;
                    }
                    if left >= 10 {
                        // Move two to the left so we revisit the left number of the new pair again.
                        cursor.prev();
                        cursor.prev();
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn reduce(&mut self) {
        loop {
            self.explode_nums();
            if !self.split_nums() {
                break;
            }
        }
    }

    fn magnitude(&self) -> i64 {
        self.root.magnitude()
    }

    fn cursor_start(&mut self) -> TreeCursor<'_> {
        TreeCursor {
            current: &mut self.root,
            position: Position::Left,
            moving_down: true,
            stack: vec![],
        }
    }
}

impl Add for SnailfishNumber {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new = SnailfishNumber {
            root: NumberNode::Pair(Box::new((self.root, rhs.root))),
        };
        new.reduce();
        new
    }
}

struct TreeCursor<'a> {
    current: &'a mut NumberNode,
    position: Position,
    moving_down: bool,
    // must only ever access the last element
    stack: Vec<(*mut NumberNode, Position)>,
}

type NodeDepth = usize;

impl<'a> TreeCursor<'a> {
    fn borrowing_clone(&mut self) -> TreeCursor<'_> {
        TreeCursor {
            current: &mut self.current,
            stack: self.stack.clone(),
            ..*self
        }
    }

    #[must_use]
    // helper for _advance
    fn _move_up(&mut self) -> Option<()> {
        let (prev, prev_pos) = self.stack.pop()?;
        self.position = prev_pos;
        self.current = unsafe { &mut *prev };
        Some(())
    }

    fn _advance(&mut self, move_right: bool) -> Option<(&mut NumberNode, NodeDepth)> {
        use NumberNode::*;
        use Position::*;

        loop {
            let current_ptr = self.current as *mut NumberNode;
            match (&mut *self.current, self.position, self.moving_down) {
                (Pair(pair), Left | Right, true) => {
                    self.stack.push((current_ptr, self.position));
                    let pair = &mut **pair;
                    let pair = unsafe { &mut *(pair as *mut (_, _)) };
                    let (left, right) = pair;

                    self.current = if self.position == Left { left } else { right };
                    self.position = if move_right { Left } else { Right };
                }
                (Number(_), _, true) => {
                    self.moving_down = false;
                    self.position = Mid; // if this node is replaced with a pair, it will be the right position
                    return Some((&mut self.current, self.stack.len()));
                }
                (Number(_), _, false) => {
                    // Numbers always have a parent in this puzzle,
                    // but I'm allowing top-level numbers as well
                    self._move_up()?;
                }
                (Pair(_), Left | Right, false) => {
                    let pos = self.position;

                    if move_right && pos == Left || (!move_right && pos == Right) {
                        self.position = Mid;
                        return Some((&mut self.current, self.stack.len()));
                    } else {
                        if self._move_up().is_none() {
                            // reached ending, reset to initial state
                            // for a cursor starting at that end
                            self.moving_down = true;
                            return None;
                        }
                    }
                }
                (Pair(_), Mid, false) => {
                    self.position = if move_right { Right } else { Left };
                    self.moving_down = true;
                }
                (Pair(_), Mid, true) => unreachable!(),
            }
        }
    }

    fn _advance_number(&mut self, move_right: bool) -> Option<&mut i64> {
        // find_map(), but manually
        loop {
            match self._advance(move_right)? {
                // This is safe, but the borrow checker can't deal with conditional return of borrows
                // even with NLLs
                // https://github.com/rust-lang/rust/issues/58910
                // The BC will keep `num` and therefore `self` borrowed across loop iterations.
                //
                // Lengthen lifetime manually
                (NumberNode::Number(num), _) => return Some(unsafe { &mut *(num as *mut i64) }),
                _ => (),
            };
        }
    }

    fn next(&mut self) -> Option<(&mut NumberNode, NodeDepth)> {
        self._advance(true)
    }

    fn prev(&mut self) -> Option<(&mut NumberNode, NodeDepth)> {
        self._advance(false)
    }

    fn next_num(&mut self) -> Option<&mut i64> {
        self._advance_number(true)
    }

    fn prev_num(&mut self) -> Option<&mut i64> {
        self._advance_number(false)
    }
}

fn day18(part: Part) {
    let input = include_str!("day18_input.txt");
    let numbers: Vec<SnailfishNumber> = input.lines().map(|num| num.parse().unwrap()).collect_vec();

    match part {
        Part::One => {
            let num = numbers.into_iter().fold1(std::ops::Add::add).unwrap();
            println!("{}", num.magnitude());
        }
        Part::Two => {
            let number_pairs = numbers.iter().enumerate().flat_map(|(i, num1)| {
                numbers
                    .iter()
                    .enumerate()
                    .filter(move |&(j, _)| i != j)
                    .map(move |(_, num2)| (num1, num2))
            });
            let max_magnitude = number_pairs
                .map(|(num1, num2)| (num1.clone() + num2.clone()).magnitude())
                .max()
                .unwrap();

            println!("{}", max_magnitude);
        }
    }
}

struct Image {
    visible_board: Board<bool>,
    outside_cells_lit: bool, // cells around the block are either all on or all off
}

fn day20(part: Part) {
    let input = include_str!("day20_input.txt");
    let (algorithm, image) = input.split_once("\n\n").unwrap();
    let algorithm = algorithm
        .as_bytes()
        .iter()
        .filter(|&&b| b != b'\n') // only necessary for test input
        .map(|&b| match b {
            b'#' => true,
            b'.' => false,
            _ => unreachable!(),
        })
        .collect_vec();

    let mut image = Image {
        visible_board: Board::from_str(image, |ch| ch == '#'),
        outside_cells_lit: false,
    };
    let apply_algorithm = |image: &Image| {
        // the algorithm can add pixels along each edge, so add a line of pixels on each side
        let old_board = &image.visible_board;
        let mut visible_board = Board::new(false, old_board.n_rows() + 2, old_board.n_cols() + 2);
        for cell in visible_board.cells() {
            let (x, y) = cell;
            let x_old = x as isize - 1;
            let y_old = y as isize - 1;
            let idx = itertools::iproduct!(y_old - 1..=y_old + 1, x_old - 1..=x_old + 1)
                .map(|(y, x)| {
                    old_board
                        .get((x as usize, y as usize))
                        .unwrap_or(&image.outside_cells_lit)
                })
                .fold(0, |num, bit| (num << 1) + *bit as usize);
            visible_board[cell] = algorithm[idx];
        }

        let outside_cells_idx = if image.outside_cells_lit { 511 } else { 0 };
        let outside_cells_lit = algorithm[outside_cells_idx];
        Image {
            visible_board,
            outside_cells_lit,
        }
    };

    let n_repetitions = match part {
        Part::One => 2,
        Part::Two => 50,
    };

    for _ in 0..n_repetitions {
        image = apply_algorithm(&image);
    }
    println!(
        "{}",
        image.visible_board.board.iter().filter(|&&lit| lit).count()
    );
}

fn day21(part: Part) {
    let input = include_str!("day21_input.txt");
    let get_num = |line: &str| parse_num(line.split(' ').last().unwrap());
    let mut positions = input.lines().map(get_num).map(|n| n - 1).collect_vec();

    match part {
        Part::One => {
            let mut scores = vec![0; positions.len()];

            let mut die_roll = 1;
            let mut n_rolls = 0;
            'game: loop {
                for (pos, score) in izip!(&mut positions, &mut scores) {
                    for _ in 0..3 {
                        n_rolls += 1;
                        *pos = (*pos + die_roll) % 10;
                        die_roll = if die_roll == 100 { 1 } else { die_roll + 1 };
                    }
                    *score += *pos + 1;

                    if *score >= 1000 {
                        break 'game;
                    }
                }
            }
            println!("{}", n_rolls * scores.into_iter().min().unwrap());
        }
        Part::Two => {
            // Manual dynamic programming.
            // Keep track of all possible states after each round for both players.
            // 20 + 10 is the max reachable score
            // I need an easier way of keeping track of all this state...
            // Maybe matrices or just a recursive solution + caching.

            let zero_poss = [[[0u64; 10]; 31]; 2];
            // scores[n_rolls][player][score][current_field]
            let mut scores = vec![zero_poss];
            scores[0][0][0][positions[0] as usize] = 1;
            scores[0][1][0][positions[1] as usize] = 1;

            let mut n_possibilities_three_die_rolls = [0; 10];
            for (d1, d2, d3) in itertools::iproduct!(1..4, 1..4, 1..4) {
                n_possibilities_three_die_rolls[d1 + d2 + d3] += 1;
            }

            // Either player must have won after this many rounds.
            // The bound can probably be lowered.
            for round in 0..22 {
                let prev_scores = &scores[round];
                let mut next_scores = zero_poss;
                for player in 0..2_usize {
                    for prev_score in 0..21_usize {
                        for prev_field in 0..10_usize {
                            for (die_roll, &n_poss) in
                                n_possibilities_three_die_rolls.iter().enumerate()
                            {
                                let next_field = (prev_field + die_roll) % 10;
                                let next_score = prev_score + (next_field + 1);
                                next_scores[player][next_score][next_field] +=
                                    prev_scores[player][prev_score][prev_field] * n_poss;
                            }
                        }
                    }
                }

                scores.push(next_scores);
            }

            let mut n_wins = [0; 2];
            for (round, results) in scores.iter().enumerate().skip(1) {
                for (player, results) in results.iter().enumerate() {
                    let previous_results = match player {
                        0 => scores[round - 1][1],
                        1 => scores[round][0],
                        _ => unreachable!(),
                    };
                    let n_not_won_other_player: u64 = previous_results[..21].iter().flatten().sum();
                    let n_won_self: u64 = results[21..].iter().flatten().sum();
                    n_wins[player] += n_not_won_other_player * n_won_self;
                }
            }
            println!("{}", std::cmp::max(n_wins[0], n_wins[1]));
        }
    }
}

#[derive(Clone, Debug)]
struct Region {
    on: bool,
    prio: usize,
    space: Space,
}

impl Region {
    fn without(&self, space: &Space) -> impl Iterator<Item = Self> + '_ {
        self.space.without(&space).map(|space| Self {
            space,
            ..self.clone()
        })
    }
}

type Range = RangeInclusive<i64>;
#[derive(Clone, Debug, PartialEq, Eq)]
struct Space {
    x: Range,
    y: Range,
    z: Range,
}

impl Space {
    fn overlap(&self, other: &Self) -> Option<Self> {
        Some(Space {
            x: Self::range_overlap(&self.x, &other.x)?,
            y: Self::range_overlap(&self.y, &other.y)?,
            z: Self::range_overlap(&self.z, &other.z)?,
        })
    }

    fn without(&self, other: &Self) -> impl Iterator<Item = Self> {
        let (below_x, above_x) = Self::ranges_above_below(&self.x, &other.x);
        let (below_y, above_y) = Self::ranges_above_below(&self.y, &other.y);
        let (below_z, above_z) = Self::ranges_above_below(&self.z, &other.z);

        [
            Space {
                x: below_x,
                ..self.clone()
            },
            Space {
                x: above_x,
                ..self.clone()
            },
            Space {
                x: other.x.clone(),
                y: below_y,
                z: self.z.clone(),
            },
            Space {
                x: other.x.clone(),
                y: above_y,
                z: self.z.clone(),
            },
            Space {
                z: below_z,
                ..other.clone()
            },
            Space {
                z: above_z,
                ..other.clone()
            },
        ]
        .into_iter()
        .filter(|space| !space.is_empty())
    }

    fn volume(&self) -> usize {
        self.x.clone().count() * self.y.clone().count() * self.z.clone().count()
    }

    fn is_empty(&self) -> bool {
        self.x.is_empty() || self.y.is_empty() || self.z.is_empty()
    }

    fn range_overlap(rg1: &Range, rg2: &Range) -> Option<Range> {
        let overlap = max(*rg1.start(), *rg2.start())..=min(*rg1.end(), *rg2.end());
        (!overlap.is_empty()).then(|| overlap)
    }

    fn ranges_above_below(rg1: &Range, rg2: &Range) -> (Range, Range) {
        (Self::range_below(rg1, rg2), Self::range_above(rg1, rg2))
    }

    fn range_below(rg1: &Range, rg2: &Range) -> Range {
        *rg1.start()..=min(*rg1.end(), *rg2.start() - 1)
    }

    fn range_above(rg1: &Range, rg2: &Range) -> Range {
        max(*rg1.start(), *rg2.end() + 1)..=*rg1.end()
    }
}

fn day22(part: Part) {
    let input = include_str!("day22_input.txt");
    let parse_range = |rg: &str| {
        let (start, end) = rg.split_once("..").unwrap();
        parse_num(start)..=parse_num(end)
    };
    let regions = input.lines().enumerate().map(|(i, line)| {
        let (setting, region) = line.split_once(' ').unwrap();
        let mut it = region.split(',').map(|rg| &rg[2..]).map(parse_range);
        let mut next = || it.next().unwrap();
        Region {
            on: setting == "on",
            prio: i,
            space: Space {
                x: next(),
                y: next(),
                z: next(),
            },
        }
    });

    let mut regions = regions.collect_vec();

    if part == Part::One {
        let inner_cube = Space {
            x: -50..=50,
            y: -50..=50,
            z: -50..=50,
        };
        regions.retain(|r| r.space.overlap(&inner_cube).is_some());
    }

    // Go through all pairs of regions (rectangular cuboids) and check if they overlap.
    // If so, take them apart into new, non-overlapping regions and the overlap region.
    // The region that came later in the input sets the values for the overlap.
    //
    // There is a lot of optimization potential here. If we were to sort the list of spaces
    // lexicographically (x,y,z), we could very easily rule out most other regions as potential
    // overlapping regions.
    for i in 0.. {
        if i >= regions.len() {
            break;
        }
        // for the `i`th region, check against all other regions after it until we find an overlap
        // and then remove that overlap by introducing new regions.
        // Repeat until no more overlaps exist.
        //
        // The regions in the growing range of 0..i don't have any pairwise overlaps afterwards.
        'other: loop {
            for j in i + 1..regions.len() {
                if let Some(overlap) = regions[i].space.overlap(&regions[j].space) {
                    let reg2 = regions.swap_remove(j);
                    let reg1 = regions.swap_remove(i);

                    regions.extend(reg1.without(&overlap));
                    regions.extend(reg2.without(&overlap));

                    let prio_reg = if reg1.prio > reg2.prio { reg1 } else { reg2 };
                    regions.push(Region {
                        space: overlap,
                        ..prio_reg.clone()
                    });
                    continue 'other;
                }
            }
            break;
        }
    }

    let n_on = regions
        .into_iter()
        .filter(|r| r.on)
        .map(|r| r.space.volume())
        .sum::<usize>();
    println!("{}", n_on);
}

#[derive(StructOpt)]
struct Opt {
    #[structopt(parse(try_from_str = parse_day))]
    day: u8,
    part: Part,
}

fn parse_day(day: &str) -> Result<u8, Box<dyn std::error::Error>> {
    match day.parse()? {
        day @ 1..=25 => Ok(day),
        _ => Err(format!("must be in range 1-25").into()),
    }
}

fn to_be_implemented(_: Part) {
    println!("not yet implemented")
}

fn main() {
    let opt = Opt::from_args();

    let day_fns = [
        day1,
        day2,
        day3,
        day4,
        day5,
        day6,
        day7,
        day8,
        day9,
        day10,
        day11,
        day12,
        day13,
        day14,
        day15,
        day16,
        day17,
        day18,
        day19::day19,
        day20,
        day21,
        day22,
    ];

    let day_fn = day_fns
        .get((opt.day - 1) as usize)
        .copied()
        .unwrap_or(to_be_implemented);
    day_fn(opt.part);
}
